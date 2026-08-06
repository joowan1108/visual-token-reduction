#!/usr/bin/env python

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

LAYERS = (9, 11, 13, 15)
CAMERAS = (0, 1)
GRID_SIZE = 8
LABEL_HEIGHT = 24
TITLE_HEIGHT = 32
INFERNO_STOPS = (
    (0.0, (0, 0, 4)),
    (0.25, (87, 16, 110)),
    (0.5, (188, 55, 84)),
    (0.75, (249, 142, 9)),
    (1.0, (252, 255, 164)),
)


def _inferno_palette() -> list[int]:
    palette = []
    for value in range(256):
        position = value / 255
        for (left, left_color), (right, right_color) in zip(INFERNO_STOPS, INFERNO_STOPS[1:], strict=False):
            if position <= right:
                amount = (position - left) / (right - left)
                palette.extend(
                    round(start + amount * (end - start))
                    for start, end in zip(left_color, right_color, strict=True)
                )
                break
    return palette


INFERNO_PALETTE = _inferno_palette()


def render_overlays(jsonl_path: Path, output_dir: Path) -> int:
    """Render the original selector-score overlays, one JSONL record per image."""
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


def _actual_attention_overlay(record: dict, jsonl_path: Path, peak: float | None = None) -> Image.Image:
    distribution = record["action_visual_attention_distribution"]
    if len(distribution) != GRID_SIZE**2:
        raise ValueError(f"Expected 64 attention values, got {len(distribution)}")
    if any(not math.isfinite(value) or value < 0 for value in distribution):
        raise ValueError("Attention values must be finite and non-negative")

    image_path = Path(record["image_path"])
    if not image_path.is_absolute():
        image_path = jsonl_path.parent / image_path
    with Image.open(image_path) as image:
        source = image.convert("RGBA")

    peak = max(distribution) if peak is None else peak
    if not peak:
        return source.convert("RGB")
    normalized = Image.new("L", (GRID_SIZE, GRID_SIZE))
    normalized.putdata([round(255 * value / peak) for value in distribution])
    smooth = normalized.resize(source.size, Image.Resampling.BICUBIC)
    colored = smooth.convert("P")
    colored.putpalette(INFERNO_PALETTE)
    heatmap = colored.convert("RGBA")
    heatmap.putalpha(smooth.point(lambda value: round(value * 190 / 255)))
    return Image.alpha_composite(source, heatmap).convert("RGB")


def _source_image(record: dict, jsonl_path: Path) -> Image.Image:
    image_path = Path(record["image_path"])
    if not image_path.is_absolute():
        image_path = jsonl_path.parent / image_path
    with Image.open(image_path) as image:
        return image.convert("RGB")


def _selection_overlay(record: dict, jsonl_path: Path) -> Image.Image:
    source = _source_image(record, jsonl_path).convert("RGBA")
    distribution = record["attention_distribution"]
    grid_size = math.isqrt(len(distribution))
    if not distribution or grid_size * grid_size != len(distribution):
        raise ValueError(f"Expected a square selection grid, got {len(distribution)} values")
    overlay = Image.new("RGBA", source.size, (0, 0, 0, 170))
    draw = ImageDraw.Draw(overlay)
    width, height = source.size
    for index, frequency in enumerate(distribution):
        if not math.isfinite(frequency) or not 0 <= frequency <= 1:
            raise ValueError("Selection frequencies must be finite values in [0, 1]")
        row, column = divmod(index, grid_size)
        color = (255, 244, 164, 190) if frequency >= 0.75 else (190, 55, 130, round(190 * frequency))
        draw.rectangle(
            (
                column * width // grid_size,
                row * height // grid_size,
                (column + 1) * width // grid_size,
                (row + 1) * height // grid_size,
            ),
            fill=color,
        )
    return Image.alpha_composite(source, overlay).convert("RGB")


def render_composites(jsonl_path: Path, output_dir: Path) -> int:
    """Render first-call final-step originals, Top-K masks, and expert-attention heatmaps."""
    records = [
        json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not records:
        raise ValueError("Diagnostics JSONL is empty")

    call_index = min(record["call_index"] for record in records)
    call_records = [record for record in records if record["call_index"] == call_index]
    denoising_step = max(record["denoising_step"] for record in call_records)
    selected = [
        record
        for record in call_records
        if record["denoising_step"] == denoising_step
        and record["layer"] in LAYERS
        and record["camera"] in CAMERAS
    ]
    by_batch = {}
    for record in selected:
        key = (record["batch_index"], record["layer"], record["camera"])
        if key in by_batch:
            raise ValueError(f"Duplicate diagnostic record for {key}")
        by_batch[key] = record

    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = 0
    for batch_index in sorted({record["batch_index"] for record in selected}):
        missing = [
            (layer, camera)
            for layer in LAYERS
            for camera in CAMERAS
            if (batch_index, layer, camera) not in by_batch
        ]
        if missing:
            raise ValueError(f"Missing batch {batch_index} diagnostics: {missing}")

        for layer in LAYERS:
            layer_records = [by_batch[(batch_index, layer, camera)] for camera in CAMERAS]
            originals = [_source_image(record, jsonl_path) for record in layer_records]
            selections = [_selection_overlay(record, jsonl_path) for record in layer_records]
            shared_peak = max(
                max(record["action_visual_attention_distribution"]) for record in layer_records
            )
            heatmaps = [
                _actual_attention_overlay(record, jsonl_path, shared_peak) for record in layer_records
            ]
            panel_width, panel_height = originals[0].size
            if any(panel.size != (panel_width, panel_height) for panel in originals + selections + heatmaps):
                raise ValueError("All camera images must have the same dimensions")

            row_height = panel_height + LABEL_HEIGHT
            composite = Image.new(
                "RGB",
                (len(CAMERAS) * panel_width, TITLE_HEIGHT + 3 * row_height),
                "white",
            )
            draw = ImageDraw.Draw(composite)
            draw.text(
                (4, 8),
                f"call={call_index} | flow={denoising_step} | cross layer={layer}",
                fill="black",
            )
            rows = (originals, selections, heatmaps)
            for column, camera in enumerate(CAMERAS):
                record = layer_records[column]
                labels = (
                    f"Camera {camera}",
                    f"Focus Top-K selection | mean selected={record['selected_token_count']:.1f}",
                    (
                        "action->visual | visual-branch camera share="
                        f"{record['action_visual_attention_mass']:.4f} | shared camera-relative scale"
                        if record.get("action_visual_attention_scope") == "visual_branch"
                        else "action->visual | total-prefix camera mass="
                        f"{record['action_visual_attention_mass']:.4f} | shared camera-relative scale"
                    ),
                )
                for row, panels in enumerate(rows):
                    x = column * panel_width
                    y = TITLE_HEIGHT + row * row_height
                    draw.text((x + 4, y + 4), labels[row], fill="black")
                    composite.paste(panels[column], (x, y + LABEL_HEIGHT))

            output_path = output_dir / (
                f"call_{call_index:06d}_flow_{denoising_step:03d}_cross_{layer:02d}_"
                f"batch_{batch_index:03d}.png"
            )
            if output_path.exists():
                raise FileExistsError(f"Refusing to overwrite composite: {output_path}")
            composite.save(output_path)
            rendered += 1
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(description="Render SmolVLA focus-token diagnostics.")
    parser.add_argument("jsonl", type=Path, help="Focus-token diagnostics JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--composite",
        action="store_true",
        help="Render actual expert-attention composites instead of selector-score overlays.",
    )
    args = parser.parse_args()
    if args.composite:
        print(f"Rendered {render_composites(args.jsonl, args.output_dir)} composites to {args.output_dir}")
    else:
        print(f"Rendered {render_overlays(args.jsonl, args.output_dir)} overlays to {args.output_dir}")


if __name__ == "__main__":
    main()
