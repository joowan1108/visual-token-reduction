# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import numpy as np
import torch


def normalized_jsd(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence normalized to [0, 1]."""
    midpoint = (left + right) / 2
    left_kl = torch.where(left > 0, left * (left.log() - midpoint.log()), 0).sum(dim=-1)
    right_kl = torch.where(right > 0, right * (right.log() - midpoint.log()), 0).sum(dim=-1)
    return (left_kl + right_kl) / (2 * np.log(2))


def iar_capture_metrics(capture: dict, state_context: torch.Tensor) -> dict[str, torch.Tensor]:
    """Reduce one transient IAR capture without retaining raw token attention.

    Query/layer diversity is normalized Jensen-Shannon divergence (0 identical, 1 maximally
    different). Padding is removed and each selected distribution is renormalized first.
    """
    attention = capture["attention"].float()
    contexts = capture["layer_contexts"].float()
    averaged_context = capture["averaged_context"].float()
    prefix_mask = capture["prefix_mask"].bool()
    if attention.ndim != 4 or attention.shape[0] != prefix_mask.shape[0]:
        raise ValueError("IAR attention must be [batch, layer, query, token].")
    if attention.shape[-1] != prefix_mask.shape[-1]:
        raise ValueError("IAR attention and prefix mask token dimensions must match.")
    if contexts.shape[:3] != attention.shape[:3] or averaged_context.shape[:2] != (
        attention.shape[0],
        attention.shape[2],
    ):
        raise ValueError("IAR contexts must align with captured layers and queries.")
    language_span = capture["language_token_span"]
    if language_span is None:
        raise ValueError("IAR token diagnostics require the language token span.")

    valid = prefix_mask[:, None, None]
    attention = attention * valid
    attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(attention.dtype).eps)
    token_count = prefix_mask.sum(dim=-1)[:, None, None]
    entropy = -(torch.where(attention > 0, attention * attention.log(), 0).sum(dim=-1))
    entropy = torch.where(token_count > 1, entropy / token_count.float().log(), torch.zeros_like(entropy))

    image_mask = torch.zeros_like(prefix_mask)
    for start, end in capture["visual_token_spans"]:
        image_mask[:, start:end] = True
    language_mask = torch.zeros_like(prefix_mask)
    language_mask[:, language_span[0] : language_span[1]] = True
    image_mask &= prefix_mask
    language_mask &= prefix_mask
    other_mask = prefix_mask & ~image_mask & ~language_mask
    image_mass = (attention * image_mask[:, None, None]).sum(dim=-1)
    language_mass = (attention * language_mask[:, None, None]).sum(dim=-1)
    other_mass = (attention * other_mask[:, None, None]).sum(dim=-1)

    def signature(token_mask: torch.Tensor | None = None) -> torch.Tensor:
        values = attention if token_mask is None else attention * token_mask[:, None, None]
        return values / values.sum(dim=(1, 2, 3), keepdim=True).clamp_min(torch.finfo(values.dtype).eps)

    query_diversity = [
        normalized_jsd(attention[:, layer, left], attention[:, layer, right])
        for layer in range(attention.shape[1])
        for left in range(attention.shape[2])
        for right in range(left + 1, attention.shape[2])
    ]
    layer_diversity = [
        normalized_jsd(attention[:, left, query], attention[:, right, query])
        for query in range(attention.shape[2])
        for left in range(attention.shape[1])
        for right in range(left + 1, attention.shape[1])
    ]
    empty = torch.empty(0, dtype=torch.float32)
    state_context = state_context.detach().float().cpu().flatten(1)
    return {
        "image_mass": image_mass,
        "language_mass": language_mass,
        "other_mass": other_mass,
        "normalized_entropy": entropy,
        "attention_signature": signature(),
        "visual_attention_signature": signature(image_mask),
        "language_attention_signature": signature(language_mask),
        "query_diversity": torch.cat(query_diversity) if query_diversity else empty,
        "layer_diversity": torch.cat(layer_diversity) if layer_diversity else empty,
        "layer_context_norm": contexts.norm(dim=-1),
        "layer_context_variance": contexts.var(dim=-1, unbiased=False),
        "context_norm": averaged_context.norm(dim=-1),
        "context_variance": averaged_context.var(dim=-1, unbiased=False),
        "state_context_norm": state_context.norm(dim=-1),
        "state_context_variance": state_context.var(dim=-1, unbiased=False),
    }


class AttentionMapCollector:
    """Average real action-to-visual attention probabilities and save compact NPZ files."""

    def __init__(self, output_dir: str | Path, num_layers: int, layers: list[int], flow_steps: list[int]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if any(not -num_layers <= layer < num_layers for layer in layers):
            raise ValueError("Attention-map layer index is out of range.")
        self.layers = tuple(dict.fromkeys(layer % num_layers for layer in layers))
        self.flow_steps = tuple(dict.fromkeys(flow_steps))
        self.call_index = 0
        self._images: np.ndarray | None = None
        self._task_descriptions: np.ndarray | None = None
        self._visual_token_spans: tuple[tuple[int, int], ...] = ()
        self._flow_step: int | None = None
        self._maps: dict[tuple[int, str, int], torch.Tensor] = {}
        self._scopes: dict[tuple[int, str, int], str] = {}

    @property
    def active(self) -> bool:
        return self._images is not None

    def start_call(
        self,
        images: list[torch.Tensor],
        visual_token_spans: tuple[tuple[int, int], ...],
        task_descriptions: list[str] | tuple[str, ...] | str | None = None,
    ) -> None:
        if self.active:
            raise RuntimeError("An attention-map call is already active.")
        token_counts = {end - start for start, end in visual_token_spans}
        if len(token_counts) != 1:
            raise ValueError("Attention-map cameras must have the same visual-token count.")
        stacked = torch.stack(images, dim=1).detach().float().cpu()
        self._images = (
            ((stacked + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).permute(0, 1, 3, 4, 2).numpy()
        )
        batch_size = self._images.shape[0]
        if task_descriptions is None:
            task_descriptions = [""] * batch_size
        elif isinstance(task_descriptions, str):
            task_descriptions = [task_descriptions]
        else:
            task_descriptions = list(task_descriptions)
        if len(task_descriptions) != batch_size or not all(
            isinstance(description, str) for description in task_descriptions
        ):
            raise ValueError("Task descriptions must be one string per attention-map batch item.")
        self._task_descriptions = np.asarray(task_descriptions, dtype=np.str_)
        self._visual_token_spans = visual_token_spans
        self._flow_step = None
        self._maps.clear()
        self._scopes.clear()

    def set_flow_step(self, flow_step: int) -> None:
        self._flow_step = flow_step

    def collect(
        self,
        probs: torch.Tensor,
        layer: int,
        attention_kind: str,
        original_indices: torch.Tensor,
        scope: str,
    ) -> None:
        if not self.active or self._flow_step is None or layer not in self.layers:
            return
        if original_indices.ndim == 1:
            original_indices = original_indices.expand(probs.shape[0], -1)
        if original_indices.shape != (probs.shape[0], probs.shape[-1]):
            raise ValueError("original_indices must align with the attention key dimension.")

        attention = probs.detach().float().mean(dim=(1, 2))
        camera_maps = []
        for start, end in self._visual_token_spans:
            camera_map = torch.zeros(probs.shape[0], end - start, dtype=torch.float32, device=probs.device)
            for batch_idx in range(probs.shape[0]):
                indices = original_indices[batch_idx]
                is_camera = (indices >= start) & (indices < end)
                camera_map[batch_idx].scatter_add_(
                    0, indices[is_camera] - start, attention[batch_idx, is_camera]
                )
            camera_maps.append(camera_map.cpu())

        key = (self._flow_step, attention_kind, layer)
        if key in self._maps:
            raise RuntimeError(f"Duplicate attention map for flow/kind/layer {key}.")
        self._maps[key] = torch.stack(camera_maps, dim=1)
        self._scopes[key] = scope

    def end_call(self) -> list[Path]:
        if not self.active:
            return []
        observed_steps = sorted({flow_step for flow_step, _, _ in self._maps})
        selected_steps = set()
        for flow_step in self.flow_steps:
            if flow_step >= 0:
                selected_steps.add(flow_step)
            elif observed_steps:
                try:
                    selected_steps.add(observed_steps[flow_step])
                except IndexError as error:
                    raise ValueError(
                        f"Requested flow step {flow_step}, but only {len(observed_steps)} were captured."
                    ) from error

        outputs = []
        for flow_step in sorted(selected_steps):
            for attention_kind in ("cross", "self"):
                keys = [
                    (flow_step, attention_kind, layer)
                    for layer in self.layers
                    if (flow_step, attention_kind, layer) in self._maps
                ]
                if not keys:
                    continue
                maps = torch.stack([self._maps[key] for key in keys]).mean(dim=0).numpy()
                output = self.output_dir / (
                    f"call_{self.call_index:06d}_flow_{flow_step:03d}_{attention_kind}.npz"
                )
                if output.exists():
                    raise FileExistsError(f"Refusing to overwrite attention map: {output}")
                np.savez_compressed(
                    output,
                    images=self._images,
                    task_descriptions=self._task_descriptions,
                    attention_maps=maps,
                    attention_mass=maps.sum(axis=-1),
                    layers=np.asarray([key[2] for key in keys], dtype=np.int64),
                    flow_step=np.asarray(flow_step, dtype=np.int64),
                    attention_kind=np.asarray(attention_kind),
                    attention_scope=np.asarray(self._scopes[keys[0]]),
                )
                outputs.append(output)

        self._images = None
        self._task_descriptions = None
        self._visual_token_spans = ()
        self._flow_step = None
        self._maps.clear()
        self._scopes.clear()
        self.call_index += 1
        return outputs
