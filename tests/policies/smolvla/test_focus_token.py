# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.smolvlm_with_expert import (
    SmolVLMWithExpertModel,
    _select_visual_tokens,
)


def test_focus_token_config_preserves_dense_default_and_validates_sparse_shape():
    assert SmolVLAConfig().focus_token_keep_ratio == 1.0

    with pytest.raises(ValueError, match="keep_ratio"):
        SmolVLAConfig(focus_token_keep_ratio=0)
    with pytest.raises(ValueError, match="16 VLM/expert layers"):
        SmolVLAConfig(focus_token_keep_ratio=0.5, focus_token_start_layer=0, num_vlm_layers=8)
    with pytest.raises(ValueError, match="compile_model=False"):
        SmolVLAConfig(focus_token_keep_ratio=0.5, compile_model=True)


def test_select_visual_tokens_uses_independent_ceil_budgets_and_restores_order():
    query = torch.ones(2, 1, 1, 1)
    keys = torch.arange(10, dtype=torch.float32).view(1, 10, 1, 1).expand(2, -1, -1, -1)
    values = keys.clone()
    mask = torch.ones(2, 1, 10, dtype=torch.bool)
    mask[1, :, 6:9] = False

    selected_keys, selected_values, selected_mask = _select_visual_tokens(
        query, keys, values, mask, ((1, 5), (6, 9)), 0.5
    )

    torch.testing.assert_close(selected_keys[0, :, 0, 0], torch.tensor([0, 3, 4, 5, 7, 8, 9.0]))
    torch.testing.assert_close(selected_values, selected_keys)
    assert selected_mask[0, 0].tolist() == [True] * 7
    torch.testing.assert_close(selected_keys[1, :5, 0, 0], torch.tensor([0, 3, 4, 5, 9.0]))
    assert selected_mask[1, 0].tolist() == [True, True, True, True, True, False, False]


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
