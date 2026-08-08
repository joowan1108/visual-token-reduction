#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import logging
import math
from pprint import pformat

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.configs.rewards import RewardModelConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.transforms import ImageTransforms
from lerobot.utils.constants import ACTION, IMAGENET_STATS, OBS_PREFIX, REWARD

from .dataset_metadata import LeRobotDatasetMetadata
from .lerobot_dataset import LeRobotDataset
from .multi_dataset import MultiLeRobotDataset
from .streaming_dataset import StreamingLeRobotDataset


def resolve_delta_timestamps(
    cfg: PreTrainedConfig | RewardModelConfig,
    ds_meta: LeRobotDatasetMetadata,
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the config.

    Args:
        cfg (PreTrainedConfig | RewardModelConfig):
            The config to read delta_indices from. Both PreTrainedConfig and
            concrete RewardModelConfig subclasses expose the
            {observation, action, reward}_delta_indices properties used below.

        ds_meta (LeRobotDatasetMetadata):
            The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None:
            A dictionary of delta_timestamps, e.g.:

            {
                "observation.state": [-0.04, -0.02, 0],
                "action": [-0.02, 0, 0.02],
            }

            Returns None if the resulting dict is empty.
    """

    delta_timestamps = {}

    subtask_delta_indices = getattr(
        cfg,
        "subtask_delta_indices",
        None,
    )

    for key in ds_meta.features:
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [
                i / ds_meta.fps
                for i in cfg.reward_delta_indices
            ]

        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [
                i / ds_meta.fps
                for i in cfg.action_delta_indices
            ]

        if (
            key.startswith(OBS_PREFIX)
            and cfg.observation_delta_indices is not None
        ):
            delta_timestamps[key] = [
                i / ds_meta.fps
                for i in cfg.observation_delta_indices
            ]

        if (
            key == "subtask_index"
            and subtask_delta_indices is not None
        ):
            delta_timestamps[key] = [
                i / ds_meta.fps
                for i in subtask_delta_indices
            ]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def make_dataset(
    cfg: TrainPipelineConfig,
) -> LeRobotDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms
    before creating a dataset.

    Args:
        cfg (TrainPipelineConfig):
            A TrainPipelineConfig config which contains a DatasetConfig
            and a PreTrainedConfig.

    Raises:
        NotImplementedError:
            The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """

    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms)
        if cfg.dataset.image_transforms.enable
        else None
    )

    if isinstance(cfg.dataset.repo_id, str):
        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            revision=cfg.dataset.revision,
        )

        delta_timestamps = resolve_delta_timestamps(
            cfg.trainable_config,
            ds_meta,
        )

        if not cfg.dataset.streaming:
            dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                video_backend=cfg.dataset.video_backend,
                return_uint8=True,
                depth_output_unit=cfg.dataset.depth_output_unit,
                tolerance_s=cfg.tolerance_s,
            )
        else:
            dataset = StreamingLeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                max_num_shards=cfg.num_workers,
                tolerance_s=cfg.tolerance_s,
                return_uint8=True,
            )

    else:
        raise NotImplementedError(
            "The MultiLeRobotDataset isn't supported for now."
        )

        dataset = MultiLeRobotDataset(
            cfg.dataset.repo_id,
            # TODO(aliberts): add proper support for multi dataset
            # delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=cfg.dataset.video_backend,
        )

        logging.info(
            "Multiple datasets were provided. "
            "Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index, indent=2)}"
        )

    # ---------------------------------------------------------
    # Apply ImageNet statistics to RGB camera observations.
    #
    # Some LeRobot datasets do not store statistics for video/image
    # features in meta/stats.json. In that case camera_keys can contain:
    #
    #   observation.images.image
    #   observation.images.image2
    #
    # while dataset.meta.stats does not contain those keys.
    #
    # Therefore initialize the dictionary before assigning mean/std.
    # ---------------------------------------------------------
    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            if key in dataset.meta.depth_keys:
                continue  # Exclude depth keys from ImageNet stats

            # FIX:
            # Avoid:
            # KeyError: 'observation.images.image'
            dataset.meta.stats.setdefault(key, {})

            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(
                    stats,
                    dtype=torch.float32,
                )

    return dataset


def make_train_eval_datasets(
    cfg: TrainPipelineConfig,
) -> tuple[LeRobotDataset | MultiLeRobotDataset, LeRobotDataset | None]:
    """Create train and optional eval datasets by splitting episodes based on eval_split.

    The last ceil(n_episodes * eval_split) episodes per task are held out
    for evaluation.

    Task grouping is determined from frame-level ``task_index``.
    This supports datasets whose meta/episodes parquet does not contain
    a ``tasks`` column.
    """

    full_dataset = make_dataset(cfg)

    if cfg.dataset.eval_split == 0.0:
        return full_dataset, None

    # ---------------------------------------------------------
    # Determine which episode indices are being used.
    # ---------------------------------------------------------
    base_episodes = (
        full_dataset.episodes
        if full_dataset.episodes is not None
        else list(range(full_dataset.num_episodes))
    )

    base_episode_set = set(base_episodes)

    # ---------------------------------------------------------
    # Build episode -> task_index mapping from the frame-level
    # Hugging Face dataset.
    #
    # Our dataset contains:
    #
    #   episode_index
    #   task_index
    #
    # but meta/episodes does NOT contain:
    #
    #   tasks
    #
    # Therefore don't use:
    #
    #   full_dataset.meta.episodes["tasks"]
    # ---------------------------------------------------------
    hf_dataset = full_dataset.hf_dataset

    if "episode_index" not in hf_dataset.column_names:
        raise ValueError(
            "Dataset does not contain frame-level 'episode_index'. "
            f"Available columns: {hf_dataset.column_names}"
        )

    if "task_index" not in hf_dataset.column_names:
        raise ValueError(
            "Dataset does not contain frame-level 'task_index'. "
            f"Available columns: {hf_dataset.column_names}"
        )

    frame_episode_indices = hf_dataset["episode_index"]
    frame_task_indices = hf_dataset["task_index"]

    episode_to_task: dict[int, int] = {}

    for ep_idx, task_idx in zip(
        frame_episode_indices,
        frame_task_indices,
        strict=True,
    ):
        ep_idx = int(ep_idx)
        task_idx = int(task_idx)

        if ep_idx not in base_episode_set:
            continue

        # LIBERO task_index should be constant inside one episode.
        # We only need the first frame to determine the episode's task.
        if ep_idx not in episode_to_task:
            episode_to_task[ep_idx] = task_idx

    # ---------------------------------------------------------
    # Ensure every selected episode has a task_index.
    # ---------------------------------------------------------
    missing_episodes = [
        ep_idx
        for ep_idx in base_episodes
        if ep_idx not in episode_to_task
    ]

    if missing_episodes:
        raise ValueError(
            "Could not determine task_index for some episodes. "
            f"Missing episodes: {missing_episodes[:20]}"
            + (
                f" ... ({len(missing_episodes)} total)"
                if len(missing_episodes) > 20
                else ""
            )
        )

    # ---------------------------------------------------------
    # Group episodes by high-level LIBERO task_index.
    # ---------------------------------------------------------
    task_to_episodes: dict[int, list[int]] = {}

    for ep_idx in base_episodes:
        task_idx = episode_to_task[ep_idx]
        task_to_episodes.setdefault(task_idx, []).append(ep_idx)

    # ---------------------------------------------------------
    # Per-task train/eval split.
    #
    # Example:
    #
    # task 0: 50 episodes, eval_split=0.1
    #     -> train 45
    #     -> eval 5
    #
    # This prevents some LIBERO tasks from disappearing entirely
    # from the evaluation split.
    # ---------------------------------------------------------
    train_episodes: list[int] = []
    eval_episodes: list[int] = []

    for task_idx, eps in sorted(task_to_episodes.items()):
        n_eval = math.ceil(len(eps) * cfg.dataset.eval_split)

        # Make sure at least one training episode remains.
        if n_eval >= len(eps):
            raise ValueError(
                f"Task {task_idx} has only {len(eps)} episodes, "
                f"but eval_split={cfg.dataset.eval_split} would leave "
                "no training episode."
            )

        train_eps = eps[: len(eps) - n_eval]
        eval_eps = eps[len(eps) - n_eval :]

        train_episodes.extend(train_eps)
        eval_episodes.extend(eval_eps)

        logging.info(
            f"Task {task_idx}: "
            f"{len(eps)} total -> "
            f"{len(train_eps)} train / "
            f"{len(eval_eps)} eval"
        )

    if not train_episodes:
        raise ValueError(
            f"eval_split={cfg.dataset.eval_split} leaves "
            f"0 training episodes from {len(base_episodes)} total."
        )

    logging.info(
        f"Train/eval split: "
        f"{len(train_episodes)} train, "
        f"{len(eval_episodes)} eval "
        f"(eval_split={cfg.dataset.eval_split}, "
        f"{len(task_to_episodes)} tasks)"
    )

    # Sort to keep deterministic episode ordering.
    train_episodes = sorted(train_episodes)
    eval_episodes = sorted(eval_episodes)

    # ---------------------------------------------------------
    # Resolve delta timestamps.
    # ---------------------------------------------------------
    delta_timestamps = resolve_delta_timestamps(
        cfg.trainable_config,
        full_dataset.meta,
    )

    train_image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms)
        if cfg.dataset.image_transforms.enable
        else None
    )

    # ---------------------------------------------------------
    # Create train dataset.
    # ---------------------------------------------------------
    train_dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=train_episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=train_image_transforms,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
        return_uint8=True,
        tolerance_s=cfg.tolerance_s,
    )

    # ---------------------------------------------------------
    # Create eval dataset.
    # ---------------------------------------------------------
    eval_dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=eval_episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=None,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
        return_uint8=True,
        tolerance_s=cfg.tolerance_s,
    )

    # ---------------------------------------------------------
    # Apply ImageNet stats.
    #
    # Camera keys may not exist in meta/stats.json, so initialize
    # them first with setdefault().
    # ---------------------------------------------------------
    if cfg.dataset.use_imagenet_stats:
        for ds in (train_dataset, eval_dataset):
            for key in ds.meta.camera_keys:
                if key in ds.meta.depth_keys:
                    continue

                # Important:
                # meta/stats.json may not contain image statistics.
                ds.meta.stats.setdefault(key, {})

                for stats_type, stats in IMAGENET_STATS.items():
                    ds.meta.stats[key][stats_type] = torch.tensor(
                        stats,
                        dtype=torch.float32,
                    )

    return train_dataset, eval_dataset