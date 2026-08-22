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

import argparse
import math
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

LABEL_HEIGHT = 24
TITLE_HEIGHT = 40


def _jet(values: np.ndarray) -> np.ndarray:
    red = np.clip(1.5 - np.abs(4 * values - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * values - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * values - 1), 0, 1)
    return np.stack((red, green, blue), axis=-1)


def _overlay(image: np.ndarray, attention: np.ndarray, peak: float) -> Image.Image:
    grid_size = math.isqrt(attention.size)
    if grid_size * grid_size != attention.size:
        raise ValueError(f"Expected a square visual-token grid, got {attention.size} values.")
    normalized = (
        attention.reshape(grid_size, grid_size) / peak
        if peak
        else np.zeros_like(attention.reshape(grid_size, grid_size))
    )
    heatmap = Image.fromarray((_jet(normalized) * 255).astype(np.uint8), mode="RGB").resize(
        (image.shape[1], image.shape[0]), Image.Resampling.BILINEAR
    )
    return Image.blend(Image.fromarray(image, mode="RGB"), heatmap, 0.55)


def render_attention_maps(input_path: Path, output_dir: Path) -> int:
    files = [input_path] if input_path.is_file() else sorted(input_path.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No attention-map NPZ files found in {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = 0
    for npz_path in files:
        with np.load(npz_path, allow_pickle=False) as data:
            images = data["images"]
            task_descriptions = (
                data["task_descriptions"].tolist()
                if "task_descriptions" in data.files
                else [""] * images.shape[0]
            )
            maps = data["attention_maps"]
            masses = data["attention_mass"]
            layers = data["layers"].tolist()
            camera_keys = (
                np.atleast_1d(data["camera_keys"]).tolist()
                if "camera_keys" in data.files
                else [str(camera_idx) for camera_idx in range(images.shape[1])]
            )
            queries = np.atleast_1d(data["queries"]).tolist() if "queries" in data.files else ["<unknown>"]
            kind = str(data["attention_kind"])
            scope = str(data["attention_scope"])
            flow_step = int(data["flow_step"])
        if images.shape[:2] != maps.shape[:2] or maps.shape != masses.shape + (maps.shape[-1],):
            raise ValueError(f"Image/map shapes do not align in {npz_path}")
        if len(camera_keys) != images.shape[1]:
            raise ValueError(f"Camera keys do not align with images in {npz_path}")

        for batch_idx in range(images.shape[0]):
            peak = float(maps[batch_idx].max())
            layer_label = ",".join(map(str, layers))
            query_label = ",".join(map(str, queries))
            height, width = images.shape[2:4]
            canvas_width = images.shape[1] * width
            canvas = Image.new("RGB", (canvas_width, TITLE_HEIGHT + 2 * (height + LABEL_HEIGHT)), "white")
            draw = ImageDraw.Draw(canvas)
            task = task_descriptions[batch_idx] or "<unknown>"
            title = "\n".join(textwrap.wrap(f"Task: {task}", width=max(20, canvas_width // 7))[:2])
            draw.multiline_text((4, 4), title, fill="black", spacing=2)
            for camera_idx in range(images.shape[1]):
                x = camera_idx * width
                source = Image.fromarray(images[batch_idx, camera_idx], mode="RGB")
                overlay = _overlay(images[batch_idx, camera_idx], maps[batch_idx, camera_idx], peak)
                draw.text((x + 4, TITLE_HEIGHT + 4), f"camera={camera_idx} | original", fill="black")
                canvas.paste(source, (x, TITLE_HEIGHT + LABEL_HEIGHT))
                y = TITLE_HEIGHT + height + LABEL_HEIGHT
                draw.multiline_text(
                    (x + 4, y + 1),
                    f"{kind}/{scope} camera={camera_keys[camera_idx]} query={query_label} "
                    f"layer={layer_label} flow={flow_step}\n"
                    f"mass={masses[batch_idx, camera_idx]:.6g} raw_visual_peak={peak:.6g} "
                    "normalization=relative_spatial_peak",
                    fill="black",
                    spacing=0,
                )
                canvas.paste(overlay, (x, y + LABEL_HEIGHT))

            output = output_dir / f"{npz_path.stem}_batch_{batch_idx:03d}.png"
            if output.exists():
                raise FileExistsError(f"Refusing to overwrite attention overlay: {output}")
            canvas.save(output)
            rendered += 1
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(description="Render SmolVLA attention-map NPZ files.")
    parser.add_argument("input", type=Path, help="An NPZ file or attention_maps_rank0 directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(f"Rendered {render_attention_maps(args.input, args.output_dir)} overlays to {args.output_dir}")


if __name__ == "__main__":
    main()
