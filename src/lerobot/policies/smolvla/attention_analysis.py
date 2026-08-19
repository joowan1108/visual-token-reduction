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
            ((stacked + 1) * 127.5)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(0, 1, 3, 4, 2)
            .numpy()
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
            camera_map = torch.zeros(
                probs.shape[0], end - start, dtype=torch.float32, device=probs.device
            )
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
