#!/usr/bin/env python
"""Validate and hash one immutable Experiment M target episode without rewriting it."""

import argparse
import hashlib
import json
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STATE

DEFAULT_TARGET_REPO = "sungkyunner/record-test_20260826_210214"
DEFAULT_TARGET_REVISION = "295e6def6cb4df454f58894caea10c15446dc4e4"
DEFAULT_TARGET_EPISODE = 0
MATCHED_SOURCE_DATASET = "Cache-SCA/Isaaclab-so101_11task_baseCaP_3300epi_10fps"
MATCHED_SOURCE_REVISION = "09a0376348f60be89edcbc0eb76c3e26b5f3b094"
MATCHED_SOURCE_EPISODE = 170
TARGET_TASK = "Pick up the red block and place it on the blue dish."
CAMERA_KEYS = ("observation.images.left_wrist", "observation.images.top")
JOINT_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)


def dataset_content_manifest(root: Path, relative_paths: list[Path] | None = None) -> dict:
    files = []
    paths = root.rglob("*") if relative_paths is None else (root / path for path in relative_paths)
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
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
    if source.fps != 10 or source.num_episodes != 1 or len(source) == 0:
        raise ValueError(
            "Target selection must be exactly one nonempty episode at 10 FPS; "
            f"got {source.num_episodes=} {len(source)=} {source.fps=}."
        )
    if list(source.meta.tasks.index) != [TARGET_TASK]:
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
        ):
            raise ValueError(f"Unexpected target camera feature {key}: {feature!r}.")


def validate_target_provenance(path: Path, repo_id: str, revision: str, episode: int) -> None:
    with path.open(encoding="utf-8") as stream:
        provenance = json.load(stream)
    expected = {
        "target_repo": repo_id,
        "target_revision": revision,
        "target_episode": episode,
        "visual_match_confirmed": True,
        "matched_source_dataset": MATCHED_SOURCE_DATASET,
        "matched_source_revision": MATCHED_SOURCE_REVISION,
        "matched_source_episode": MATCHED_SOURCE_EPISODE,
    }
    actual = {key: provenance.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"Target provenance does not match this run: expected {expected!r}, got {actual!r}.")


def prepare_target_dataset(
    output: Path,
    repo_id: str,
    revision: str,
    episode: int,
    visual_match_confirmed: bool,
) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite target provenance {output}.")
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("Target revision must be an immutable 40-character lowercase commit SHA.")
    if episode < 0:
        raise ValueError("Target episode must be nonnegative.")
    if not visual_match_confirmed:
        raise ValueError("Visual source/target layout match must be confirmed before preparation.")

    source = LeRobotDataset(
        repo_id,
        revision=revision,
        episodes=[episode],
        video_backend="pyav",
    )
    validate_target_contract(source)
    selected_paths = [Path(path) for path in source.reader.get_episodes_file_paths()]
    content_manifest = dataset_content_manifest(source.root, selected_paths)
    provenance = {
        "target_repo": repo_id,
        "target_revision": revision,
        "target_episode": episode,
        "visual_match_confirmed": True,
        "matched_source_dataset": MATCHED_SOURCE_DATASET,
        "matched_source_revision": MATCHED_SOURCE_REVISION,
        "matched_source_episode": MATCHED_SOURCE_EPISODE,
        "selected_frames": len(source),
        "target_fps": source.fps,
        "target_task": TARGET_TASK,
        "features": [OBS_STATE, ACTION, *CAMERA_KEYS],
        "conversions": [],
        "selected_content_sha256": content_manifest["tree_sha256"],
        "content_manifest": content_manifest,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(provenance, stream, indent=2)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_TARGET_REPO)
    parser.add_argument("--revision", default=DEFAULT_TARGET_REVISION)
    parser.add_argument("--episode", type=int, default=DEFAULT_TARGET_EPISODE)
    parser.add_argument("--visual-match-confirmed", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-provenance", type=Path)
    args = parser.parse_args()
    if args.verify_provenance is not None:
        validate_target_provenance(args.verify_provenance, args.repo_id, args.revision, args.episode)
    elif args.output is not None:
        prepare_target_dataset(
            args.output,
            args.repo_id,
            args.revision,
            args.episode,
            args.visual_match_confirmed,
        )
    else:
        parser.error("one of --output or --verify-provenance is required")


if __name__ == "__main__":
    main()
