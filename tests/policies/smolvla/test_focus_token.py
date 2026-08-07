# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import json
from contextlib import nullcontext
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from experiments.focus_token.visualize_focus_tokens import render_composites, render_overlays
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.smolvlm_with_expert import (
    SmolVLMWithExpertModel,
    _per_query_visual_topk_mask,
    _select_visual_tokens,
)


def test_focus_token_config_preserves_dense_default_and_validates_sparse_shape():
    config = SmolVLAConfig()
    assert config.focus_token_keep_ratio == 1.0
    assert config.focus_token_diagnostics_path is None
    assert not config.focus_cascaded_attention
    assert not config.focus_channel_gate

    with pytest.raises(ValueError, match="keep_ratio"):
        SmolVLAConfig(focus_token_keep_ratio=0)
    with pytest.raises(ValueError, match="16 VLM/expert layers"):
        SmolVLAConfig(focus_token_keep_ratio=0.5, focus_token_start_layer=0, num_vlm_layers=8)
    with pytest.raises(ValueError, match="compile_model=False"):
        SmolVLAConfig(focus_token_keep_ratio=0.5, compile_model=True)
    with pytest.raises(ValueError, match="requires"):
        SmolVLAConfig(focus_channel_gate=True)


def test_per_query_global_topk_selects_exact_and_different_visual_patches():
    query = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    keys = torch.tensor([[[[4.0, 0.0]], [[3.0, 0.0]], [[0.0, 2.0]], [[0.0, 1.0]]]])
    mask = torch.ones(1, 2, 4, dtype=torch.bool)

    selected, _ = _per_query_visual_topk_mask(query, keys, mask, 0.5)

    assert selected.tolist() == [[[True, True, False, False], [False, False, True, True]]]
    assert selected.sum(dim=-1).tolist() == [[2, 2]]


def test_per_query_global_topk_uses_per_query_valid_ceil_budget():
    query = torch.ones(1, 3, 1, 2)
    keys = torch.arange(8, dtype=torch.float32).view(1, 4, 1, 2)
    mask = torch.tensor([[[True, True, True, True], [True, True, True, False], [False] * 4]])

    selected, _ = _per_query_visual_topk_mask(query, keys, mask, 0.5)

    assert selected.sum(dim=-1).tolist() == [[2, 2, 0]]
    assert not (selected & ~mask).any()


def test_select_visual_tokens_uses_independent_ceil_budgets_and_restores_order():
    query = torch.ones(2, 1, 1, 1)
    keys = torch.arange(10, dtype=torch.float32).view(1, 10, 1, 1).expand(2, -1, -1, -1)
    values = keys.clone()
    mask = torch.ones(2, 1, 10, dtype=torch.bool)
    mask[0, :, 6] = False
    mask[1, :, 6:9] = False

    diagnostics = []
    selected_keys, selected_values, selected_mask, original_indices = _select_visual_tokens(
        query, keys, values, mask, ((1, 5), (6, 9)), 0.5, diagnostics
    )

    torch.testing.assert_close(selected_keys[0, :, 0, 0], torch.tensor([0, 3, 4, 5, 8, 9.0]))
    torch.testing.assert_close(selected_values, selected_keys)
    assert original_indices[0].tolist() == [0, 3, 4, 5, 8, 9]
    assert selected_mask[0, 0].tolist() == [True] * 6
    torch.testing.assert_close(selected_keys[1, :5, 0, 0], torch.tensor([0, 3, 4, 5, 9.0]))
    assert selected_mask[1, 0].tolist() == [True, True, True, True, True, False]
    first_camera = diagnostics[0]
    assert first_camera["valid_token_count"] == 4
    assert first_camera["selected_token_count"] == 2
    assert first_camera["selected_indices"] == [2, 3]
    assert first_camera["selected_prefix_indices"] == [3, 4]
    expected_distribution = torch.softmax(torch.tensor([1.0, 2.0, 3.0, 4.0]), dim=0)
    torch.testing.assert_close(torch.tensor(first_camera["attention_distribution"]), expected_distribution)
    expected_mass = torch.softmax(torch.tensor([1.0, 2.0, 3.0, 4.0]), dim=0)[-2:].sum()
    assert first_camera["topk_attention_mass"] == pytest.approx(expected_mass.item())
    assert first_camera["topk_attention_mass"] == pytest.approx(
        sum(first_camera["attention_distribution"][index] for index in first_camera["selected_indices"])
    )
    assert diagnostics[2]["attention_distribution"][0] == 0.0
    assert sum(diagnostics[2]["attention_distribution"]) == pytest.approx(1.0)
    assert diagnostics[3]["valid_token_count"] == 0
    assert diagnostics[3]["selected_indices"] == []
    assert diagnostics[3]["attention_distribution"] == [0.0, 0.0, 0.0]


class _FakeAttention:
    def __init__(self, input_size: int, query_size: int, key_value_size: int):
        self.head_dim = 2
        self.q_proj = nn.Linear(input_size, query_size, bias=False)
        self.k_proj = nn.Linear(input_size, key_value_size, bias=False)
        self.v_proj = nn.Linear(input_size, key_value_size, bias=False)


class _FakeLayer:
    def __init__(self, input_size: int, query_size: int, key_value_size: int):
        self.input_layernorm = nn.Identity()
        self.self_attn = _FakeAttention(input_size, query_size, key_value_size)


def _make_cross_attention_model():
    model = SmolVLMWithExpertModel.__new__(SmolVLMWithExpertModel)
    nn.Module.__init__(model)
    model.num_attention_heads = 2
    model.num_key_value_heads = 1
    model.focus_token_keep_ratio = 0.5
    model.focus_token_start_layer = 8
    model.focus_cascaded_attention = False
    model.focus_channel_gate = False
    vlm_layer = _FakeLayer(4, 4, 2)
    expert_layer = _FakeLayer(4, 4, 2)
    expert_layer.self_attn.k_proj = nn.Linear(2, 2, bias=False)
    expert_layer.self_attn.v_proj = nn.Linear(2, 2, bias=False)
    return model, [[vlm_layer] * 16, [expert_layer] * 16]


def test_focus_token_layer_boundary_dense_identity_and_direct_cached_paths():
    torch.manual_seed(0)
    model, layers = _make_cross_attention_model()
    prefix = torch.randn(1, 8, 4)
    suffix = torch.randn(1, 3, 4, requires_grad=True)
    direct_mask = torch.ones(1, 11, 11, dtype=torch.bool)
    direct_positions = torch.arange(11).unsqueeze(0)
    observed_key_lengths = []

    def attention(mask, batch_size, head_dim, queries, keys, values):
        observed_key_lengths.append(keys.shape[1])
        return model.eager_attention_forward(mask, batch_size, head_dim, queries, keys, values)

    model.get_attention_interface = lambda: attention
    for layer_idx in range(1, 16, 2):
        outputs, _ = model.forward_cross_attn_layer(
            layers,
            [prefix, suffix],
            layer_idx,
            direct_positions,
            direct_mask,
            1,
            2,
            use_cache=False,
            visual_token_spans=((0, 4), (4, 8)),
        )
        assert torch.isfinite(outputs[1]).all()
        assert observed_key_lengths[-1] == (4 if layer_idx in (9, 11, 13, 15) else 8)

    outputs[1].sum().backward()
    assert suffix.grad is not None and torch.isfinite(suffix.grad).all()

    model.focus_token_keep_ratio = 1.0
    dense_with_spans, _ = model.forward_cross_attn_layer(
        layers,
        [prefix, suffix.detach()],
        9,
        direct_positions,
        direct_mask,
        1,
        2,
        use_cache=False,
        visual_token_spans=((0, 4), (4, 8)),
    )
    dense_without_spans, _ = model.forward_cross_attn_layer(
        layers, [prefix, suffix.detach()], 9, direct_positions, direct_mask, 1, 2, use_cache=False
    )
    torch.testing.assert_close(dense_with_spans[1], dense_without_spans[1], rtol=0, atol=0)

    model.focus_token_keep_ratio = 0.5
    cache_keys = torch.randn(1, 1, 8, 2)
    cache_values = torch.randn(1, 1, 8, 2)
    cache = SimpleNamespace(layers=[SimpleNamespace(keys=cache_keys, values=cache_values) for _ in range(16)])
    cached_suffix = torch.randn(1, 3, 4, requires_grad=True)
    cached_outputs, _ = model.forward_cross_attn_layer(
        layers,
        [None, cached_suffix],
        9,
        torch.arange(3).unsqueeze(0),
        torch.ones(1, 3, 8, dtype=torch.bool),
        1,
        2,
        use_cache=True,
        past_key_values=cache,
        visual_token_spans=((0, 4), (4, 8)),
    )
    actions = nn.Linear(4, 2)(cached_outputs[0])
    assert torch.isfinite(actions).all()
    actions.sum().backward()
    assert cached_suffix.grad is not None and torch.isfinite(cached_suffix.grad).all()
    assert all(layer.keys.shape[2] == 8 for layer in cache.layers)


def test_cascaded_attention_normalizes_branches_and_backpropagates_gate_and_fusion():
    torch.manual_seed(1)
    model, layers = _make_cross_attention_model()
    model.focus_cascaded_attention = True
    model.focus_channel_gate = True
    model.cascaded_fusion = nn.ModuleDict({"1": nn.Linear(8, 4)})
    model.focus_gates = nn.ModuleDict({"1": nn.Sequential(nn.Linear(4, 4), nn.SiLU(), nn.Linear(4, 4))})
    prefix = torch.randn(1, 10, 4)
    suffix = torch.randn(1, 2, 4, requires_grad=True)
    observed_masks = []

    def attention(mask, batch_size, head_dim, queries, keys, values):
        observed_masks.append(mask.detach().clone())
        return model.eager_attention_forward(mask, batch_size, head_dim, queries, keys, values)

    model.get_attention_interface = lambda: attention
    outputs, _ = model.forward_cross_attn_layer(
        layers,
        [prefix, suffix],
        1,
        torch.arange(12).unsqueeze(0),
        torch.ones(1, 12, 12, dtype=torch.bool),
        1,
        2,
        use_cache=False,
        visual_token_spans=((0, 4), (4, 8)),
    )

    # Prefix self-attention, condition branch, then visual branch.
    assert observed_masks[-2].shape[-1] == 2
    assert observed_masks[-2].all()
    assert observed_masks[-1].shape[-1] == 8
    assert observed_masks[-1].sum(dim=-1).tolist() == [[4, 4]]
    gate = torch.sigmoid(model.focus_gates["1"](suffix))
    assert gate.shape == suffix.shape
    assert torch.all((0 <= gate) & (gate <= 1))
    outputs[1].sum().backward()
    assert suffix.grad is not None and torch.isfinite(suffix.grad).all()
    assert model.cascaded_fusion["1"].weight.grad is not None
    assert model.focus_gates["1"][0].weight.grad is not None
    assert model.focus_gates["1"][2].weight.grad is not None


def test_eager_attention_zeroes_masked_probabilities_and_fully_invalid_queries():
    model, _ = _make_cross_attention_model()
    queries = torch.ones(1, 2, 2, 2)
    keys = torch.ones(1, 3, 1, 2)
    values = torch.arange(6, dtype=torch.float32).view(1, 3, 1, 2)
    mask = torch.tensor([[[True, False, True], [False, False, False]]])
    captured = []
    model._capture_action_visual_attention = lambda probs: captured.append(probs.detach())

    output = model.eager_attention_forward(mask, 1, 2, queries, keys, values)

    assert torch.equal(captured[0][..., 1], torch.zeros_like(captured[0][..., 1]))
    assert torch.equal(captured[0][:, :, 1], torch.zeros_like(captured[0][:, :, 1]))
    assert torch.equal(output[:, 1], torch.zeros_like(output[:, 1]))


def test_eager_attention_condition_padding_does_not_change_output():
    torch.manual_seed(2)
    model, _ = _make_cross_attention_model()
    queries = torch.randn(1, 1, 2, 2)
    keys = torch.randn(1, 3, 1, 2)
    values = torch.randn(1, 3, 1, 2)
    padded_keys = torch.cat([keys, torch.randn(1, 1, 1, 2)], dim=1)
    padded_values = torch.cat([values, torch.randn(1, 1, 1, 2)], dim=1)

    expected = model.eager_attention_forward(
        torch.ones(1, 1, 3, dtype=torch.bool), 1, 2, queries, keys, values
    )
    actual = model.eager_attention_forward(
        torch.tensor([[[True, True, True, False]]]), 1, 2, queries, padded_keys, padded_values
    )

    torch.testing.assert_close(actual, expected)


def test_cascaded_diagnostics_name_visual_branch_camera_share():
    model, _ = _make_cross_attention_model()
    records = [
        {
            "batch_index": 0,
            "camera": 0,
            "action_visual_attention_scope": "visual_branch",
        }
    ]
    model._focus_token_attention_diagnostics = (
        ((0, 2),),
        torch.tensor([[0, 1]]),
        records,
    )

    model._capture_action_visual_attention(torch.tensor([[[[0.25, 0.75]]]]))

    assert records[0]["action_visual_attention_scope"] == "visual_branch"
    assert records[0]["visual_branch_camera_attention_share"] == pytest.approx(1.0)
    assert "total_prefix_camera_attention_mass" not in records[0]


def test_focus_token_diagnostics_are_opt_in_and_align_calls_with_images(monkeypatch):
    model, layers = _make_cross_attention_model()
    prefix = torch.arange(32, dtype=torch.float32).view(1, 8, 4)
    suffix = torch.ones(1, 3, 4)
    path = Path("selection_metrics.jsonl")
    output = StringIO()
    saved_images = {}
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: None)
    monkeypatch.setattr(Path, "exists", lambda *args, **kwargs: False)
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: nullcontext(output))
    monkeypatch.setattr(
        Image.Image, "save", lambda image, image_path: saved_images.setdefault(image_path, image)
    )
    model.get_attention_interface = lambda: model.eager_attention_forward
    args = (
        layers,
        [prefix, suffix],
        9,
        torch.arange(11).unsqueeze(0),
        torch.ones(1, 11, 11, dtype=torch.bool),
        1,
        2,
    )

    model.forward_cross_attn_layer(*args, use_cache=False, visual_token_spans=((0, 4), (4, 8)))
    assert output.getvalue() == ""

    model.enable_focus_token_diagnostics(path, frame=12, denoising_step=3)
    images = [torch.full((1, 3, 2, 2), -1.0), torch.zeros(1, 3, 2, 2)]
    model.start_focus_token_diagnostics_call(images)
    model.forward_cross_attn_layer(*args, use_cache=False, visual_token_spans=((0, 4), (4, 8)))
    model.end_focus_token_diagnostics_call()
    model.start_focus_token_diagnostics_call(images)
    model.forward_cross_attn_layer(*args, use_cache=False, visual_token_spans=((0, 4), (4, 8)))
    model.end_focus_token_diagnostics_call()

    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert len(records) == 4
    assert records[0]["layer"] == 9
    assert records[0]["camera"] == 0
    assert records[0]["frame"] == 12
    assert records[0]["denoising_step"] == 3
    assert records[0]["call_index"] == 0
    assert records[2]["call_index"] == 1
    assert records[0]["image_path"] != records[1]["image_path"]
    assert records[0]["image_path"] != records[2]["image_path"]
    assert records[0]["valid_token_count"] == 4
    assert records[0]["selected_token_count"] == 2
    assert 0 < records[0]["topk_attention_mass"] <= 1
    assert len(records[0]["action_visual_attention_distribution"]) == records[0]["valid_token_count"]
    assert records[0]["action_visual_attention_mass"] == pytest.approx(
        sum(records[0]["action_visual_attention_distribution"])
    )
    assert records[0]["action_visual_attention_mass"] + records[1][
        "action_visual_attention_mass"
    ] == pytest.approx(1.0)
    assert records[0]["action_visual_attention_scope"] == "total_prefix"
    assert records[0]["total_prefix_camera_attention_mass"] == pytest.approx(
        records[0]["action_visual_attention_mass"]
    )
    assert all(
        value == 0
        for index, value in enumerate(records[0]["action_visual_attention_distribution"])
        if index not in records[0]["selected_indices"]
    )
    assert saved_images[Path(records[0]["image_path"])].getpixel((0, 0)) == (0, 0, 0)
    assert saved_images[Path(records[1]["image_path"])].getpixel((0, 0)) == (128, 128, 128)

    output.seek(0)
    output.truncate()
    model.focus_token_keep_ratio = 1.0
    model.enable_focus_token_diagnostics(path, frame=12, denoising_step=3)
    model.start_focus_token_diagnostics_call(images)
    model.forward_cross_attn_layer(*args, use_cache=False, visual_token_spans=((0, 4), (4, 8)))
    model.end_focus_token_diagnostics_call()

    dense_records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert len(dense_records) == 2
    assert dense_records[0]["selected_indices"] == [0, 1, 2, 3]
    assert len(dense_records[0]["action_visual_attention_distribution"]) == dense_records[0][
        "valid_token_count"
    ]
    assert dense_records[0]["action_visual_attention_mass"] + dense_records[1][
        "action_visual_attention_mass"
    ] == pytest.approx(1.0)


def test_focus_token_visualizers_preserve_selector_overlays_and_render_smooth_composite(tmp_path):
    image_path = tmp_path / "camera.png"
    Image.new("RGB", (32, 32), "black").save(image_path)
    records = []
    for layer in (9, 11, 13, 15):
        for camera in (0, 1):
            selector_distribution = [0.0] * 64
            selector_distribution[0] = 1.0
            action_distribution = [0.0] * 64
            action_distribution[camera] = 0.25
            records.append(
                {
                    "call_index": 0,
                    "batch_index": 0,
                    "camera": camera,
                    "layer": layer,
                    "denoising_step": 9,
                    "image_path": str(image_path),
                    "selected_indices": [0],
                    "selected_token_count": 1,
                    "attention_distribution": selector_distribution,
                    "action_visual_attention_mass": 0.25,
                    "action_visual_attention_distribution": action_distribution,
                }
            )
    jsonl_path = tmp_path / "attention.jsonl"
    jsonl_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    overlay_dir = tmp_path / "overlays"
    assert render_overlays(jsonl_path, overlay_dir) == 8
    assert len(list(overlay_dir.glob("*.png"))) == 8

    composite_dir = tmp_path / "composites"
    assert render_composites(jsonl_path, composite_dir) == 4
    with Image.open(next(composite_dir.glob("*.png"))) as composite:
        assert composite.size == (64, 200)
        panel = composite.crop((0, 168, 32, 200))
        assert len(panel.getcolors(maxcolors=32 * 32)) > 8
