#!/usr/bin/env python
"""Validate and hash the pinned same-rig target episode without rewriting it."""

import argparse
import hashlib
import json
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STATE

SOURCE_REPO = "sungkyunner/record-test_20260825_225339"
SOURCE_REVISION = "97e2c1d4d49607210d1e63d46db2a43b530bdf89"
SOURCE_EPISODE = 0
SOURCE_TASK = "Pick up the red block and place it on the blue dish."
EXPECTED_FRAMES = 300
CAMERA_KEYS = ("observation.images.left_wrist", "observation.images.top")
JOINT_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)


def dataset_content_manifest(root: Path) -> dict:
    files = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": digest.hexdigest(),
                "size": path.stat().st_size,
            }
        )

    tree = hashlib.sha256()
    for file in files:
        tree.update(f"{file['path']}\0{file['sha256']}\n".encode())
    return {"algorithm": "sha256(path\\0sha256\\n)", "tree_sha256": tree.hexdigest(), "files": files}


def validate_target_contract(source: LeRobotDataset) -> None:
    if source.fps != 10 or source.num_episodes != 1 or len(source) != EXPECTED_FRAMES:
        raise ValueError(
            f"Pinned target must be one {EXPECTED_FRAMES}-frame episode at 10 FPS; "
            f"got {source.num_episodes=} {len(source)=} {source.fps=}."
        )
    if list(source.meta.tasks.index) != [SOURCE_TASK]:
        raise ValueError(f"Unexpected task table: {list(source.meta.tasks.index)!r}.")

    for key in (OBS_STATE, ACTION):
        feature = source.meta.features.get(key, {})
        if feature.get("dtype") != "float32" or tuple(feature.get("shape") or ()) != (6,):
            raise ValueError(f"{key} must be one float32 six-joint vector, got {feature!r}.")
        if tuple(feature.get("names") or ()) != JOINT_NAMES:
            raise ValueError(f"Unexpected {key} joint order: {feature.get('names')!r}.")
        gripper_min = float(source.meta.stats[key]["min"][-1])
        gripper_max = float(source.meta.stats[key]["max"][-1])
        if not 0 <= gripper_min <= gripper_max <= 100:
            raise ValueError(f"{key} gripper is not in the current [0, 100] convention.")

    for key in CAMERA_KEYS:
        feature = source.meta.features.get(key, {})
        video_info = feature.get("info") or {}
        if (
            feature.get("dtype") != "video"
            or tuple(feature.get("shape") or ()) != (480, 640, 3)
            or video_info.get("video.fps") != 10
            or video_info.get("video.codec") != "av1"
        ):
            raise ValueError(f"Unexpected pinned camera feature {key}: {feature!r}.")


def prepare_target_dataset(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite target provenance {output}.")

    source = LeRobotDataset(
        SOURCE_REPO,
        revision=SOURCE_REVISION,
        episodes=[SOURCE_EPISODE],
        video_backend="pyav",
    )
    validate_target_contract(source)
    provenance = {
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "source_episode": SOURCE_EPISODE,
        "source_frames": EXPECTED_FRAMES,
        "source_fps": 10,
        "source_task": SOURCE_TASK,
        "features": [OBS_STATE, ACTION, *CAMERA_KEYS],
        "conversions": [],
        "content_manifest": dataset_content_manifest(source.root),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(provenance, stream, indent=2)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    prepare_target_dataset(parser.parse_args().output)


if __name__ == "__main__":
    main()
