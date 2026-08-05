#!/usr/bin/env python

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw


def render_overlays(jsonl_path: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = 0
    with jsonl_path.open(encoding="utf-8") as records:
        for line_number, line in enumerate(records, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            distribution = record["attention_distribution"]
            grid_size = math.isqrt(len(distribution))
            if not distribution or grid_size * grid_size != len(distribution):
                raise ValueError(
                    f"Line {line_number}: expected a non-empty square token grid, got {len(distribution)} tokens"
                )
            if any(not math.isfinite(value) or value < 0 for value in distribution):
                raise ValueError(
                    f"Line {line_number}: attention_distribution must be finite and non-negative"
                )

            image_path = Path(record["image_path"])
            if not image_path.is_absolute():
                image_path = jsonl_path.parent / image_path
            with Image.open(image_path) as image:
                source = image.convert("RGBA")

            width, height = source.size
            overlay = Image.new("RGBA", source.size)
            draw = ImageDraw.Draw(overlay)
            peak = max(distribution)
            for index, value in enumerate(distribution):
                row, column = divmod(index, grid_size)
                bounds = (
                    column * width // grid_size,
                    row * height // grid_size,
                    (column + 1) * width // grid_size - 1,
                    (row + 1) * height // grid_size - 1,
                )
                if peak:
                    draw.rectangle(bounds, fill=(255, 32, 0, round(180 * value / peak)))

            for index in record["selected_indices"]:
                if not isinstance(index, int) or not 0 <= index < len(distribution):
                    raise ValueError(f"Line {line_number}: selected patch index {index!r} is out of range")
                row, column = divmod(index, grid_size)
                draw.rectangle(
                    (
                        column * width // grid_size,
                        row * height // grid_size,
                        (column + 1) * width // grid_size - 1,
                        (row + 1) * height // grid_size - 1,
                    ),
                    outline=(255, 255, 0, 255),
                    width=max(1, min(width, height) // 112),
                )

            output_path = output_dir / (
                f"call_{record['call_index']:06d}_batch_{record['batch_index']:03d}_"
                f"camera_{record['camera']:02d}_layer_{record['layer']:02d}_"
                f"step_{record['denoising_step']:02d}_line_{line_number:06d}.png"
            )
            if output_path.exists():
                raise FileExistsError(f"Refusing to overwrite overlay: {output_path}")
            Image.alpha_composite(source, overlay).convert("RGB").save(output_path)
            rendered += 1
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay focus-token attention on saved model-input images.")
    parser.add_argument("jsonl", type=Path, help="Focus-token diagnostics JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(f"Rendered {render_overlays(args.jsonl, args.output_dir)} overlays to {args.output_dir}")


if __name__ == "__main__":
    main()
