#!/usr/bin/env python
"""Canonicalize the pinned public target episode into the SO-101 policy interface."""

import argparse
import hashlib
import json
import tempfile
from copy import deepcopy
from pathlib import Path

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STATE

SOURCE_REPO = "skkuprism/test_pick_red_place_blue_50epi_10fps"
SOURCE_REVISION = "e19331e77f477a4be16f7c2884250ed6f491e048"
SOURCE_EPISODE = 0
SOURCE_TASK = "pick up the red block and place it on the blue dish"
TARGET_TASK = "Pick up the red block and place it on the blue dish."
TARGET_REPO = "local/domain-arithmetic-so101-target-canonical-ep0"
EXPECTED_FRAMES = 329
CAMERA_KEYS = ("observation.images.left_wrist", "observation.images.top")
FEATURE_KEYS = (OBS_STATE, ACTION, *CAMERA_KEYS)


def dataset_content_manifest(root: Path) -> dict:
    files = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative_path = path.relative_to(root).as_posix()
        if not path.is_file() or relative_path == "target_preparation.json":
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        files.append({"path": relative_path, "sha256": digest.hexdigest(), "size": path.stat().st_size})

    tree = hashlib.sha256()
    for file in files:
        tree.update(f"{file['path']}\0{file['sha256']}\n".encode())
    return {"algorithm": "sha256(path\\0sha256\\n)", "tree_sha256": tree.hexdigest(), "files": files}


def canonicalize_joint_vector(vector: torch.Tensor) -> torch.Tensor:
    """Convert only the gripper from old [-100, 100] to current [0, 100]."""
    if vector.shape != (6,) or not vector.is_floating_point() or not torch.isfinite(vector).all():
        raise ValueError(f"Expected one finite floating six-joint vector, got {vector.shape} {vector.dtype}.")
    if not -100 <= vector[-1] <= 100:
        raise ValueError(f"Old-convention gripper is outside [-100, 100]: {vector[-1].item()}.")
    converted = vector.clone()
    converted[-1] = (converted[-1] + 100) / 2
    return converted


def image_for_writer(image: torch.Tensor, feature: dict) -> torch.Tensor:
    """Return decoded uint8 RGB in the feature's declared HWC layout."""
    expected = tuple(feature["shape"])
    if not isinstance(image, torch.Tensor) or image.dtype != torch.uint8 or len(expected) != 3:
        raise ValueError(f"Expected a decoded uint8 RGB tensor for feature {feature}.")
    if tuple(image.shape) == (expected[2], expected[0], expected[1]):
        image = image.permute(1, 2, 0)
    if tuple(image.shape) != expected:
        raise ValueError(f"Decoded image shape {tuple(image.shape)} does not match {expected}.")
    return image.contiguous()


def prepare_target_dataset(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing target dataset {output}.")

    source = LeRobotDataset(
        SOURCE_REPO,
        revision=SOURCE_REVISION,
        episodes=[SOURCE_EPISODE],
        return_uint8=True,
    )
    if source.fps != 10 or source.num_episodes != 1 or len(source) != EXPECTED_FRAMES:
        raise ValueError(
            f"Pinned target must be one {EXPECTED_FRAMES}-frame episode at 10 FPS; "
            f"got {source.num_episodes=} {len(source)=} {source.fps=}."
        )
    missing = set(FEATURE_KEYS) - set(source.meta.features)
    if missing:
        raise ValueError(f"Pinned target is missing required features: {sorted(missing)}.")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temp:
        target_root = Path(temp) / "dataset"
        target = LeRobotDataset.create(
            repo_id=TARGET_REPO,
            fps=source.fps,
            features={key: deepcopy(source.meta.features[key]) for key in FEATURE_KEYS},
            root=target_root,
            robot_type=source.meta.robot_type,
            use_videos=True,
        )
        for index in range(EXPECTED_FRAMES):
            frame = source[index]
            if frame["task"] != SOURCE_TASK:
                raise ValueError(f"Unexpected task at frame {index}: {frame['task']!r}.")
            state = canonicalize_joint_vector(frame[OBS_STATE])
            action = canonicalize_joint_vector(frame[ACTION])
            target.add_frame(
                {
                    OBS_STATE: state,
                    ACTION: action,
                    CAMERA_KEYS[0]: image_for_writer(
                        frame[CAMERA_KEYS[0]], source.meta.features[CAMERA_KEYS[0]]
                    ),
                    CAMERA_KEYS[1]: image_for_writer(
                        frame[CAMERA_KEYS[1]], source.meta.features[CAMERA_KEYS[1]]
                    ),
                    "task": TARGET_TASK,
                }
            )
        target.save_episode(parallel_encoding=False)
        target.finalize()
        if target.num_episodes != 1 or target.num_frames != EXPECTED_FRAMES:
            raise RuntimeError(
                f"Prepared target has {target.num_episodes} episodes and {target.num_frames} frames."
            )
        content_manifest = dataset_content_manifest(target_root)
        (target_root / "target_preparation.json").write_text(
            json.dumps(
                {
                    "source_repo": SOURCE_REPO,
                    "source_revision": SOURCE_REVISION,
                    "source_episode": SOURCE_EPISODE,
                    "source_frames": EXPECTED_FRAMES,
                    "source_task": SOURCE_TASK,
                    "target_repo": TARGET_REPO,
                    "target_task": TARGET_TASK,
                    "features": list(FEATURE_KEYS),
                    "conversions": {
                        f"{OBS_STATE}[-1]": "(old + 100) / 2",
                        f"{ACTION}[-1]": "(old + 100) / 2",
                    },
                    "content_manifest": content_manifest,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        target_root.rename(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    prepare_target_dataset(parser.parse_args().output)


if __name__ == "__main__":
    main()
