#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

"""
SmolVLA:

[Paper](https://huggingface.co/papers/2506.01844)

Designed by Hugging Face.

Install smolvla extra dependencies:
```bash
pip install -e ".[smolvla]"
```

Example of finetuning the smolvla pretrained model (`smolvla_base`):
```bash
lerobot-train \
--policy.path=lerobot/smolvla_base \
--dataset.repo_id=<USER>/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of finetuning a smolVLA. SmolVLA is composed of a pretrained VLM,
and an action expert.
```bash
lerobot-train \
--policy.type=smolvla \
--dataset.repo_id=<USER>/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of using the smolvla pretrained model outside LeRobot training framework:
```python
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
```

"""

import logging
import math
import time
from collections import deque
from typing import TypedDict, Unpack

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.device_utils import resolve_safetensors_device
from lerobot.utils.import_utils import require_package

from ..common.flow_matching import euler_integrate, sample_noise, sample_time_beta
from ..common.vla_utils import (
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    pad_vector,
    resize_with_pad,
)
from ..pretrained import PreTrainedPolicy, RolloutEpisodeFailure
from ..rtc.modeling_rtc import RTCProcessor
from ..utils import populate_queues
from .configuration_smolvla import SmolVLAConfig
from .smolvlm_with_expert import SmolVLMWithExpertModel

ATOMIC_SKILLS = ("pick", "place", "push", "turn", "open", "close")
logger = logging.getLogger(__name__)


class AtomicPlannerEpisodeFailure(RolloutEpisodeFailure):
    """The preregistered planner failure rule ended the current episode."""


def parse_atomic_planner_output(raw_output: str) -> int:
    if not isinstance(raw_output, str) or (skill := raw_output.strip()) not in ATOMIC_SKILLS:
        raise ValueError("Atomic planner output must be exactly one allowed skill word.")
    return ATOMIC_SKILLS.index(skill)


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None
    current_skill: Tensor | None
    current_phase: Tensor | None
    atomic_skill_id: Tensor | None


def _list_safetensor_keys(model_file: str, map_location: str) -> set[str]:
    from safetensors import safe_open

    with safe_open(model_file, framework="pt", device=resolve_safetensors_device(map_location)) as handle:
        return set(handle.keys())


def _load_safetensor_model(model, model_file: str, map_location: str, strict: bool):
    from safetensors.torch import load_model

    return load_model(model, model_file, strict=strict, device=resolve_safetensors_device(map_location))


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with smolvla which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by smolvla to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


class SmolVLAPolicy(PreTrainedPolicy):
    """Wrapper class around VLAFlowMatching model to train and run inference within LeRobot."""

    config_class = SmolVLAConfig
    name = "smolvla"

    def supports_rtc(self) -> bool:
        return True

    def __init__(
        self,
        config: SmolVLAConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """

        require_package("transformers", extra="smolvla")
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = VLAFlowMatching(config, rtc_processor=self.rtc_processor)
        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        self._current_skill = None
        self._pending_skill = None
        self._last_transition_prediction = None
        self._current_phase = None
        self._pending_phase = None
        self._last_phase_prediction = None
        self._atomic_planner_skill = None
        self._atomic_planner_consecutive_failures = 0
        self.atomic_planner_history = []

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Lets create processor if the config provided
        # If RTC is not enabled - we still can track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            # In case of calling init_rtc_processor after the model is created
            # We need to set the rtc_processor to the model
            # During the normal initialization process the model is not created yet
            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def get_optim_params(self) -> dict:
        return self.parameters()

    def _get_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        # TODO: Check if this for loop is needed.
        # Context: In fact, self.queues contains only ACTION field, and in inference, we don't have action in the batch
        # In the case of offline inference, we have the action in the batch
        # that why without the k != ACTION check, it will raise an error because we are trying to stack
        # on an empty container.
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        current_phase = kwargs.get("current_phase")
        images, img_masks = self.prepare_images(batch, current_phase=current_phase)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        actions = self.model.sample_actions(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            noise=noise,
            task_descriptions=batch.get("task"),
            **kwargs,
        )
        if self._skill_linking_enabled():
            actions, transition_logits = actions
            transition_logits = transition_logits.clone()
            transition_logits[:, 0] = torch.finfo(transition_logits.dtype).min
            self._pending_skill = transition_logits.argmax(dim=-1)
            self._last_transition_prediction = self._pending_skill.clone()
        elif self._phase_masking_enabled():
            actions, phase_logits = actions
            self._pending_phase = phase_logits.argmax(dim=-1)
            self._last_phase_prediction = self._pending_phase.clone()

        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return actions

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        return batch

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        if (
            self._skill_linking_enabled()
            or self._phase_masking_enabled()
            or self._atomic_planner_enabled()
            or self._atomic_classifier_enabled()
        ):
            raise RuntimeError(
                "Stateful skill/phase/planner prediction supports synchronous select_action only."
            )
        self.eval()

        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        actions = self._get_action_chunk(batch, noise, **kwargs)
        return actions

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """

        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if self._check_get_actions_condition():
            atomic_skill_id = kwargs.get("atomic_skill_id")
            if self._atomic_planner_enabled() or self._atomic_classifier_enabled():
                if atomic_skill_id is not None:
                    raise ValueError(
                        "Do not manually pass `atomic_skill_id` while atomic skill prediction is enabled."
                    )
                atomic_skill_id = (
                    self.replan_atomic_skill_classifier(batch)
                    if self._atomic_classifier_enabled()
                    else self.replan_atomic_skill(batch)
                )
            current_skill = None
            current_phase = None
            if self._skill_linking_enabled():
                self._ensure_skill_state(batch[OBS_STATE].shape[0], batch[OBS_STATE].device)
                self._apply_pending_skill()
                current_skill = self._current_skill
            elif self._phase_masking_enabled():
                self._ensure_phase_state(batch[OBS_STATE].shape[0], batch[OBS_STATE].device)
                self._apply_pending_phase()
                current_phase = self._current_phase

            actions = self._get_action_chunk(
                batch,
                noise,
                current_skill=current_skill,
                current_phase=current_phase,
                atomic_skill_id=atomic_skill_id,
            )

            # `self.predict_action_chunk` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        return self._queues[ACTION].popleft()

    def _check_get_actions_condition(self) -> bool:
        return len(self._queues[ACTION]) == 0

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _skill_linking_enabled(self) -> bool:
        return getattr(self.config, "skill_linking_enabled", False)

    def _phase_masking_enabled(self) -> bool:
        return getattr(self.config, "phase_camera_masking_enabled", False)

    def _atomic_planner_enabled(self) -> bool:
        return getattr(self.config, "atomic_planner_enabled", False)

    def _atomic_classifier_enabled(self) -> bool:
        return getattr(self.config, "atomic_classifier_enabled", False)

    def replan_atomic_skill_classifier(self, batch: dict[str, Tensor]) -> Tensor:
        if batch[OBS_STATE].shape[0] != 1:
            raise ValueError("Atomic classifier evaluation requires sequential batch-size-1 rollouts.")
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        previous_skill = (
            len(ATOMIC_SKILLS) if self._atomic_planner_skill is None else self._atomic_planner_skill
        )
        started = time.perf_counter()
        logits = self.model.classify_atomic_skill(
            images,
            img_masks,
            batch[f"{OBS_LANGUAGE_TOKENS}"],
            batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"],
            state,
            torch.tensor([previous_skill], device=state.device),
        )
        skill_id = int(logits.argmax(dim=-1).item())
        if skill_id != self._atomic_planner_skill:
            self._queues[ACTION].clear()
        self._atomic_planner_skill = skill_id
        skill = ATOMIC_SKILLS[skill_id]
        logger.info("Atomic classifier predicted skill: %s", skill)
        self.atomic_planner_history.append(
            {
                "skill": skill,
                "parse_failure": False,
                "logits": logits[0].float().cpu().tolist(),
                "latency_s": time.perf_counter() - started,
            }
        )
        return torch.tensor([skill_id], dtype=torch.long, device=batch[OBS_STATE].device)

    def _atomic_planner_prompt(self, task: str) -> str:
        previous = "none" if self._atomic_planner_skill is None else ATOMIC_SKILLS[self._atomic_planner_skill]
        history = [entry["skill"] for entry in self.atomic_planner_history if not entry["parse_failure"]]
        history_text = ", ".join(history) if history else "none"
        return (
            "Choose the next skill based on the current images.\n\n"
            "pick: the gripper is not holding the target and must grasp it\n"
            "place: the gripper is holding the target and must release it at the destination\n"
            "push: move an object by pushing without grasping\n"
            "turn: rotate a knob or switch\n"
            "open: open a drawer, cabinet, or appliance\n"
            "close: close a drawer, cabinet, or appliance\n\n"
            "Output exactly one word from:\n"
            "pick, place, push, turn, open, close\n\n"
            f"Task: {task}\nPrevious skill: {previous}\nExecuted skill history: {history_text}\n"
            "Answer:"
        )

    def replan_atomic_skill(self, batch: dict[str, Tensor]) -> Tensor:
        """Run one Think–Act-style planning interval using the policy's own frozen VLM."""
        batch_size = batch[OBS_STATE].shape[0]
        if batch_size != 1:
            raise ValueError("Frozen atomic planner evaluation requires sequential batch-size-1 rollouts.")
        task = batch.get("task", [""])
        task = task[0] if isinstance(task, (list, tuple)) else task
        if not isinstance(task, str) or not task:
            raise ValueError("Frozen atomic planner requires one non-empty task instruction.")

        images = []
        for key in self.config.image_features:
            if key not in batch:
                continue
            image = batch[key]
            image = image[:, -1] if image.ndim == 5 else image
            images.append(image[0])
        if len(images) != 2:
            raise ValueError("Frozen atomic planner requires exactly the main and wrist camera images.")
        prompt = self._atomic_planner_prompt(task)
        started = time.perf_counter()
        raw_output = self.model.vlm_with_expert.generate_atomic_planner_output(images, prompt)
        try:
            skill_id = parse_atomic_planner_output(raw_output)
        except ValueError as error:
            logger.warning("Atomic planner raw output: %r", raw_output)
            self._atomic_planner_consecutive_failures += 1
            self.atomic_planner_history.append(
                {
                    "prompt": prompt,
                    "raw_output": raw_output,
                    "parse_failure": True,
                    "latency_s": time.perf_counter() - started,
                }
            )
            if self._atomic_planner_skill is None or self._atomic_planner_consecutive_failures >= 2:
                raise AtomicPlannerEpisodeFailure(str(error)) from error
            return torch.tensor(
                [self._atomic_planner_skill], dtype=torch.long, device=batch[OBS_STATE].device
            )

        self._atomic_planner_consecutive_failures = 0
        if skill_id != self._atomic_planner_skill:
            self._queues[ACTION].clear()
        self._atomic_planner_skill = skill_id
        logger.info("Atomic planner predicted skill: %s", ATOMIC_SKILLS[skill_id])
        self.atomic_planner_history.append(
            {
                "prompt": prompt,
                "raw_output": raw_output,
                "skill": ATOMIC_SKILLS[skill_id],
                "parse_failure": False,
                "latency_s": time.perf_counter() - started,
            }
        )
        return torch.tensor([skill_id], dtype=torch.long, device=batch[OBS_STATE].device)

    def _atomic_batch_contract(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        labels = batch.get("subtask_index")
        labels_is_pad = batch.get("subtask_index_is_pad")
        if labels is None or labels_is_pad is None:
            raise KeyError("Atomic data handling requires `subtask_index` and `subtask_index_is_pad`.")
        if labels.ndim != 2 or labels.shape != labels_is_pad.shape:
            raise ValueError("Atomic subtask labels and padding mask must be aligned rank-2 tensors.")
        anchor = 1 if self._atomic_classifier_enabled() else 0
        if labels.shape[1] != self.config.chunk_size + anchor or labels_is_pad[:, anchor].any():
            raise ValueError("Atomic subtask windows must cover the chunk and have an unpadded anchor.")

        mapping = torch.tensor(self.config.atomic_subtask_to_skill, device=labels.device)
        valid = ~labels_is_pad.bool()
        if ((labels[valid] < 0) | (labels[valid] >= mapping.numel())).any():
            raise ValueError("Atomic subtask index is outside the frozen mapping vocabulary.")
        safe_labels = labels.long().clamp(0, mapping.numel() - 1)
        skills = mapping[safe_labels]
        skills = skills[:, anchor:]
        labels_is_pad = labels_is_pad[:, anchor:].bool()
        current_skill = skills[:, 0]
        boundary_is_pad = labels_is_pad | (skills != current_skill[:, None])
        return current_skill, boundary_is_pad

    def _atomic_classifier_targets(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        current_skill, _ = self._atomic_batch_contract(batch)
        labels = batch["subtask_index"]
        labels_is_pad = batch["subtask_index_is_pad"].bool()
        mapping = torch.tensor(self.config.atomic_subtask_to_skill, device=labels.device)
        previous_skill = torch.full_like(current_skill, len(ATOMIC_SKILLS))
        has_previous = ~labels_is_pad[:, 0]
        previous_skill[has_previous] = mapping[labels[has_previous, 0].long()]
        return current_skill, previous_skill

    def _ensure_phase_state(self, batch_size: int, device: torch.device) -> None:
        if torch.is_tensor(self._current_phase):
            if self._current_phase.shape == (batch_size,) and self._current_phase.device == device:
                return
        self._current_phase = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._pending_phase = None

    def _apply_pending_phase(self) -> None:
        if self._pending_phase is None:
            return
        self._current_phase = self._pending_phase
        self._pending_phase = None

    def _build_phase_targets(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        phase_index = batch.get("phase_index")
        phase_is_pad = batch.get("phase_index_is_pad")
        if phase_index is None or phase_is_pad is None:
            raise KeyError(
                "Phase-aware masking requires per-frame `phase_index` labels; "
                "semantic `subtask_index` alone cannot define MOVE/INTERACT boundaries."
            )
        if phase_index.ndim != 2 or phase_is_pad.ndim != 2 or phase_index.shape != phase_is_pad.shape:
            raise ValueError("`phase_index` and `phase_index_is_pad` must be aligned rank-2 tensors.")
        if phase_index.shape[1] != 2:
            raise ValueError("Phase-aware masking requires phase labels at offsets [0, n_action_steps].")

        pad = phase_is_pad.bool()
        if pad[:, 0].any():
            raise ValueError("Current phase cannot be padded.")
        valid_labels = phase_index[~pad]
        if not torch.all((valid_labels == -1) | (valid_labels == 1)):
            raise ValueError("Long-VLA phase labels must use moving=-1 and interaction=+1.")
        phases = (phase_index > 0).long()
        return phases[:, 0], phases[:, 1], ~pad[:, 1]

    def _skill_start_index(self) -> int:
        return int(self.config.skill_linking_num_skills)

    def _skill_done_index(self) -> int:
        return self._skill_start_index() + 1

    def _ensure_skill_state(self, batch_size: int, device: torch.device) -> None:
        if torch.is_tensor(self._current_skill):
            if self._current_skill.shape == (batch_size,) and self._current_skill.device == device:
                return
        self._current_skill = torch.full(
            (batch_size,), self._skill_start_index(), dtype=torch.long, device=device
        )
        self._pending_skill = None

    def _apply_pending_skill(self) -> None:
        if self._pending_skill is None:
            return
        update = (self._pending_skill > 0) & (self._pending_skill < self._skill_start_index())
        self._current_skill = torch.where(update, self._pending_skill, self._current_skill)
        self._pending_skill = None

    def _validate_skill_transition_batch(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        horizon = self.config.n_action_steps
        subtask_index = batch["subtask_index"]
        subtask_index_is_pad = batch["subtask_index_is_pad"]
        if subtask_index.ndim != 2 or subtask_index_is_pad.ndim != 2:
            raise ValueError(
                "Skill linking requires rank-2 `subtask_index` and `subtask_index_is_pad` tensors."
            )
        if subtask_index.shape != subtask_index_is_pad.shape:
            raise ValueError("`subtask_index` and `subtask_index_is_pad` must have the same shape.")
        if subtask_index.shape[1] <= horizon:
            raise ValueError(
                f"Skill linking requires at least H+1 subtask labels; got shape {tuple(subtask_index.shape)} for H={horizon}."
            )
        frame_index = batch["frame_index"]
        if frame_index.ndim > 1:
            frame_index = frame_index[:, 0]
        if frame_index.ndim != 1 or frame_index.shape[0] != subtask_index.shape[0]:
            raise ValueError("`frame_index` must align with the batch dimension of `subtask_index`.")

        labels = subtask_index.long()
        pad = subtask_index_is_pad.bool()
        num_skills = self.config.skill_linking_num_skills
        current = labels[:, 0]
        if pad[:, 0].any():
            raise ValueError("`subtask_index[:, 0]` cannot be padded for skill linking.")
        if ((current < 1) | (current >= num_skills)).any():
            raise ValueError(f"Observed current semantic IDs must be in [1, {num_skills - 1}].")
        valid_labels = labels[~pad]
        if ((valid_labels < 1) | (valid_labels >= num_skills)).any():
            raise ValueError(f"Observed semantic IDs must be in [1, {num_skills - 1}] when not padded.")
        return labels, pad, frame_index.long()

    def _build_skill_transition_targets(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        horizon = self.config.n_action_steps
        subtask_index, subtask_index_is_pad, frame_index = self._validate_skill_transition_batch(batch)

        start = self._skill_start_index()
        done = self._skill_done_index()
        current_skill = subtask_index[:, 0].clone()
        current_skill[frame_index == 0] = start

        future = subtask_index[:, 1 : horizon + 1].long()
        future_pad = subtask_index_is_pad[:, 1 : horizon + 1]
        terminated = future_pad.any(dim=1)
        distinct = (future != current_skill[:, None]) & ~future_pad
        first_distinct = distinct.to(dtype=torch.int64).argmax(dim=1)
        target = torch.full_like(current_skill, start)
        has_distinct = distinct.any(dim=1)
        target[has_distinct] = future.gather(1, first_distinct[:, None]).squeeze(1)[has_distinct]
        target[terminated] = done
        if (target == 0).any():
            raise ValueError("Transition target class 0 is reserved and invalid.")
        return current_skill, target

    @classmethod
    def _load_as_safetensor(cls, model, model_file: str, map_location: str, strict: bool):
        if getattr(model.config, "atomic_sgmoe_enabled", False):
            from safetensors.torch import load_file

            from lerobot.policies.utils import log_model_loading_keys

            checkpoint = load_file(model_file, device=resolve_safetensors_device(map_location))
            current = model.state_dict()
            atomic_keys = {key for key in current if ".mlp.shared_expert." in key}
            checkpoint_atomic_keys = {key for key in checkpoint if ".mlp.shared_expert." in key}
            if checkpoint_atomic_keys and checkpoint_atomic_keys != atomic_keys:
                raise ValueError("Partial Atomic SG-MoE checkpoint is not allowed.")

            promoted_from_dense = not checkpoint_atomic_keys
            if not checkpoint_atomic_keys:
                promoted = dict(checkpoint)
                for target_key in atomic_keys:
                    source_key = target_key.replace(".mlp.shared_expert.", ".mlp.")
                    if source_key not in checkpoint:
                        raise ValueError(f"Dense checkpoint is missing action FFN key: {source_key}")
                    promoted[target_key] = checkpoint[source_key]
                    for expert_id in range(6):
                        promoted[target_key.replace(".shared_expert.", f".skill_experts.{expert_id}.")] = (
                            checkpoint[source_key].clone()
                        )
                    promoted.pop(source_key)
                checkpoint = promoted

            allowed_missing = (
                {key for key in current if ".atomic_router." in key} if promoted_from_dense else set()
            )
            classifier_keys = {
                key
                for key in current
                if key.startswith("model.atomic_classifier.")
                or key.startswith("model.atomic_previous_skill_embedding.")
            }
            present_classifier_keys = classifier_keys & set(checkpoint)
            if present_classifier_keys and present_classifier_keys != classifier_keys:
                raise ValueError("Partial atomic classifier checkpoint is not allowed.")
            if not present_classifier_keys:
                allowed_missing |= classifier_keys
            incompatible = model.load_state_dict(checkpoint, strict=False)
            if set(incompatible.missing_keys) - allowed_missing:
                raise ValueError(f"Missing non-router keys in Atomic checkpoint: {incompatible.missing_keys}")
            if incompatible.unexpected_keys:
                raise ValueError(f"Unexpected Atomic checkpoint keys: {incompatible.unexpected_keys}")
            log_model_loading_keys(incompatible.missing_keys, incompatible.unexpected_keys)
            return model

        skill_linking_enabled = getattr(model.config, "skill_linking_enabled", False)
        phase_masking_enabled = getattr(model.config, "phase_camera_masking_enabled", False)
        if not skill_linking_enabled and not phase_masking_enabled:
            return super()._load_as_safetensor(model, model_file, map_location, strict)
        if strict:
            return super()._load_as_safetensor(model, model_file, map_location, strict)

        from lerobot.policies.utils import log_model_loading_keys

        if skill_linking_enabled:
            allowed_missing_prefixes = ("model.skill_embedding.", "model.transition_head.")
            feature_name = "skill-linking"
        else:
            allowed_missing_prefixes = ("model.phase_embedding.", "model.phase_head.")
            feature_name = "phase-masking"
        current = model.state_dict()
        checkpoint_keys = _list_safetensor_keys(model_file, map_location)
        feature_keys = [
            key for key in current if any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
        ]
        present_feature_keys = [key for key in feature_keys if key in checkpoint_keys]
        if present_feature_keys and len(present_feature_keys) != len(feature_keys):
            missing_feature_keys = [key for key in feature_keys if key not in checkpoint_keys]
            raise ValueError(
                f"Partial {feature_name} checkpoint is not allowed: missing {missing_feature_keys}"
            )
        unexpected = sorted(set(checkpoint_keys) - set(current))
        if unexpected:
            raise ValueError(f"Unexpected keys in checkpoint: {unexpected}")

        missing_keys, unexpected_keys = _load_safetensor_model(model, model_file, map_location, strict=False)
        if unexpected_keys:
            raise ValueError(f"Unexpected keys after load: {unexpected_keys}")
        if any(
            not any(key.startswith(prefix) for prefix in allowed_missing_prefixes) for key in missing_keys
        ):
            raise ValueError(f"Missing non-{feature_name} keys after load: {missing_keys}")
        log_model_loading_keys(missing_keys, unexpected_keys)
        return model

    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"
    ) -> dict[str, Tensor]:
        """Do a full training forward pass to compute the loss.

        Args:
            batch: Training batch containing observations and actions.
            noise: Optional noise tensor for flow matching.
            time: Optional time tensor for flow matching.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
        """
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        current_skill = None
        auxiliary_target = None
        auxiliary_valid = None
        auxiliary_name = None
        current_phase = None
        if self._skill_linking_enabled():
            current_skill, auxiliary_target = self._build_skill_transition_targets(batch)
            auxiliary_valid = torch.ones_like(auxiliary_target, dtype=torch.bool)
            auxiliary_name = "transition"
        elif self._phase_masking_enabled():
            current_phase, auxiliary_target, auxiliary_valid = self._build_phase_targets(batch)
            auxiliary_name = "phase"

        images, img_masks = self.prepare_images(batch, current_phase=current_phase)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        if self._atomic_classifier_enabled():
            target, previous_skill = self._atomic_classifier_targets(batch)
            logits = self.model.classify_atomic_skill(
                images, img_masks, lang_tokens, lang_masks, state, previous_skill
            )
            losses = F.cross_entropy(logits, target, reduction="none")
            loss_dict = {
                "atomic_classifier_loss": losses.mean().item(),
                "atomic_classifier_accuracy": (logits.argmax(dim=-1) == target).float().mean().item(),
                "loss": losses.mean().item(),
            }
            return (losses if reduction == "none" else losses.mean()), loss_dict
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")
        atomic_skill_id = None
        if self.config.atomic_data_enabled:
            atomic_skill_id, boundary_is_pad = self._atomic_batch_contract(batch)
            actions_is_pad = (
                boundary_is_pad if actions_is_pad is None else actions_is_pad.bool() | boundary_is_pad
            )
        loss_dict = {}
        model_output = self.model.forward(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            actions,
            noise,
            time,
            current_skill=current_skill,
            current_phase=current_phase,
            atomic_skill_id=atomic_skill_id,
        )
        auxiliary_logits = None
        if auxiliary_name is not None:
            losses, auxiliary_logits = model_output
            if auxiliary_name == "transition":
                auxiliary_logits = auxiliary_logits.clone()
                auxiliary_logits[:, 0] = torch.finfo(auxiliary_logits.dtype).min
        else:
            losses = model_output
        original_action_dim = self.config.action_feature.shape[0]
        losses = losses[:, :, :original_action_dim]
        loss_dict["losses_after_forward"] = losses.clone().mean().item()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone().mean().item()

        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone().mean().item()

        if auxiliary_name is None:
            if reduction == "none":
                # Return per-sample losses (B,) by averaging over valid (time, action) entries
                if actions_is_pad is None:
                    per_sample_loss = losses.mean(dim=(1, 2))
                else:
                    num_valid = ((~actions_is_pad).sum(dim=1) * losses.shape[-1]).clamp_min(1)
                    per_sample_loss = losses.sum(dim=(1, 2)) / num_valid
                loss_dict["loss"] = per_sample_loss.mean().item()
                return per_sample_loss, loss_dict
            else:
                # Default: return scalar mean loss over valid (time, action) entries
                if actions_is_pad is None:
                    loss = losses.mean()
                else:
                    num_valid = ((~actions_is_pad).sum() * losses.shape[-1]).clamp_min(1)
                    loss = losses.sum() / num_valid
                loss_dict["loss"] = loss.item()
                return loss, loss_dict

        class_weights = (
            getattr(self.config, "skill_transition_class_weights", None)
            if self._skill_linking_enabled()
            else None
        )
        auxiliary_weight = (
            None
            if class_weights is None
            else torch.tensor(class_weights, dtype=torch.float32, device=auxiliary_logits.device)
        )
        auxiliary_lambda = (
            self.config.skill_transition_loss_weight
            if self._skill_linking_enabled()
            else self.config.phase_loss_weight
        )
        logged_logits = auxiliary_logits[:, 1:] if auxiliary_name == "transition" else auxiliary_logits
        loss_dict[f"{auxiliary_name}_logits_mean"] = logged_logits.float().mean().item()
        auxiliary_losses = F.cross_entropy(
            auxiliary_logits.float(),
            auxiliary_target,
            weight=auxiliary_weight,
            reduction="none",
        )
        auxiliary_losses = auxiliary_losses * auxiliary_valid.to(dtype=auxiliary_losses.dtype)

        if reduction == "none":
            if actions_is_pad is None:
                flow_loss = losses.mean(dim=(1, 2))
            else:
                num_valid = ((~actions_is_pad).sum(dim=1) * losses.shape[-1]).clamp_min(1)
                flow_loss = losses.sum(dim=(1, 2)) / num_valid
            per_sample_loss = flow_loss + auxiliary_lambda * auxiliary_losses
            loss_dict["flow_loss"] = flow_loss.mean().item()
            loss_dict[f"{auxiliary_name}_loss"] = (
                auxiliary_losses.sum() / auxiliary_valid.sum().clamp_min(1)
            ).item()
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict

        if actions_is_pad is None:
            flow_loss = losses.mean()
        else:
            num_valid = ((~actions_is_pad).sum() * losses.shape[-1]).clamp_min(1)
            flow_loss = losses.sum() / num_valid
        auxiliary_loss = auxiliary_losses.sum() / auxiliary_valid.sum().clamp_min(1)
        loss = flow_loss + auxiliary_lambda * auxiliary_loss
        loss_dict["flow_loss"] = flow_loss.item()
        loss_dict[f"{auxiliary_name}_loss"] = auxiliary_loss.item()
        loss_dict["loss"] = loss.item()
        return loss, loss_dict

    def prepare_images(self, batch, current_phase: Tensor | None = None):
        """Apply SmolVLA preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                # SmolVLA stores the target as (width, height); the shared helper expects (height, width).
                img = resize_with_pad(
                    img,
                    self.config.resize_imgs_with_padding[1],
                    self.config.resize_imgs_with_padding[0],
                    pad_value=0,
                )

            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            if self._phase_masking_enabled():
                if current_phase is None:
                    raise ValueError("Phase-aware masking requires current_phase.")
                phase = current_phase.to(device=device).bool()
                if key == self.config.phase_static_camera_key:
                    mask = mask & ~phase
                elif key == self.config.phase_wrist_camera_key:
                    mask = mask & phase
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)
        return images, img_masks

    def _pi_aloha_decode_state(self, state):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def prepare_state(self, batch):
        """Pad state"""
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    def _get_default_peft_targets(self) -> dict[str, any]:
        """Return default PEFT target modules for SmolVLA fine-tuning."""
        common_projections = (
            "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
        )
        target_modules = rf"(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.({common_projections}))"
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
        }

    def _validate_peft_config(self, peft_config) -> None:
        """Validate PEFT configuration for SmolVLA."""
        super()._validate_peft_config(peft_config)
        if not self.config.load_vlm_weights:
            import logging

            logging.warning(
                "Training SmolVLA from scratch using PEFT. This is unlikely to yield good results. "
                "Set `load_vlm_weights=True` to fine-tune the existing policy."
            )


def pad_tensor(tensor, max_len, pad_value=0):
    """
    Efficiently pads a tensor along sequence dimension to match max_len.

    Args:
        tensor (torch.Tensor): Shape (B, L, ...) or (B, L).
        max_len (int): Fixed sequence length.
        pad_value (int/float): Value for padding.

    Returns:
        torch.Tensor: Shape (B, max_len, ...) or (B, max_len).
    """
    b, d = tensor.shape[:2]

    # Create a padded tensor of max_len and copy the existing values
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, :d] = tensor  # Efficient in-place copy

    return padded_tensor


class SkillTransitionHead(nn.Module):
    def __init__(self, vlm_dim: int, skill_dim: int, num_classes: int):
        super().__init__()
        self.visual_proj = nn.Linear(vlm_dim, skill_dim)
        self.task_proj = nn.Linear(vlm_dim, skill_dim)
        self.skill_proj = nn.Linear(skill_dim, skill_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(skill_dim * 3),
            nn.Linear(skill_dim * 3, skill_dim),
            nn.GELU(),
            nn.Linear(skill_dim, num_classes),
        )

    @staticmethod
    def _pool(tokens: Tensor, mask: Tensor, query: Tensor) -> Tensor:
        if not mask.bool().any(dim=1).all():
            raise ValueError("Transition attention pooling requires at least one valid token per sample.")
        scores = torch.einsum("bd,btd->bt", query, tokens) / math.sqrt(tokens.shape[-1])
        scores = scores.masked_fill(~mask.bool(), torch.finfo(scores.dtype).min)
        return torch.einsum("bt,btd->bd", scores.softmax(dim=-1), tokens)

    def forward(
        self,
        visual_tokens: Tensor,
        visual_mask: Tensor,
        task_tokens: Tensor,
        task_mask: Tensor,
        skill: Tensor,
    ) -> Tensor:
        dtype = self.visual_proj.weight.dtype
        visual = self.visual_proj(visual_tokens.detach().to(dtype=dtype))
        task = self.task_proj(task_tokens.detach().to(dtype=dtype))
        query = self.skill_proj(skill.to(dtype=dtype))
        logits = self.classifier(
            torch.cat(
                [self._pool(visual, visual_mask, query), self._pool(task, task_mask, query), query],
                dim=-1,
            )
        ).float()
        logits[:, 0] = torch.finfo(logits.dtype).min
        return logits


class AtomicSkillClassifier(nn.Module):
    def __init__(self, vlm_dim: int, hidden_dim: int, state_dim: int):
        super().__init__()
        self.visual_proj = nn.Linear(vlm_dim, hidden_dim)
        self.task_proj = nn.Linear(vlm_dim, hidden_dim)
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(ATOMIC_SKILLS)),
        )

    def forward(
        self,
        visual_tokens: Tensor,
        visual_mask: Tensor,
        task_tokens: Tensor,
        task_mask: Tensor,
        state: Tensor,
        previous_skill: Tensor,
    ) -> Tensor:
        dtype = self.visual_proj.weight.dtype
        visual = self.visual_proj(visual_tokens.detach().to(dtype=dtype))
        task = self.task_proj(task_tokens.detach().to(dtype=dtype))
        previous_skill = previous_skill.to(dtype=dtype)
        return self.classifier(
            torch.cat(
                [
                    SkillTransitionHead._pool(visual, visual_mask, previous_skill),
                    SkillTransitionHead._pool(task, task_mask, previous_skill),
                    self.state_proj(state.detach().to(dtype=dtype)),
                    previous_skill,
                ],
                dim=-1,
            )
        ).float()


class VLAFlowMatching(nn.Module):
    """
    SmolVLA

    [Paper]()

    Designed by Hugging Face.
    ┌──────────────────────────────┐
    │                 actions      │
    │                    ▲         │
    │ ┌─────────┐      ┌─|────┐    │
    │ |         │────► │      │    │
    │ |         │ kv   │      │    │
    │ |         │────► │Action│    │
    │ |   VLM   │cache │Expert│    |
    │ │         │────► |      │    │
    │ │         │      │      │    │
    │ └▲──▲───▲─┘      └───▲──┘    |
    │  │  |   |            │       |
    │  |  |   |          noise     │
    │  │  │ state                  │
    │  │ language tokens           │
    │  image(s)                    │
    └──────────────────────────────┘
    """

    def __init__(self, config: SmolVLAConfig, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config

        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=self.config.vlm_model_name,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            load_vlm_weights=self.config.load_vlm_weights,
            attention_mode=self.config.attention_mode,
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
            focus_token_keep_ratio=self.config.focus_token_keep_ratio,
            focus_token_start_layer=self.config.focus_token_start_layer,
            focus_token_diagnostics_path=self.config.focus_token_diagnostics_path,
            focus_cascaded_attention=self.config.focus_cascaded_attention,
            focus_channel_gate=self.config.focus_channel_gate,
            attention_map=self.config.attention_map,
            attention_map_output_dir=self.config.attention_map_output_dir,
            attention_map_layers=self.config.attention_map_layers,
            attention_map_flow_steps=self.config.attention_map_flow_steps,
            atomic_sgmoe_enabled=self.config.atomic_sgmoe_enabled,
            device=self.config.device if self.config.device is not None else "auto",
        )
        self.state_proj = nn.Linear(
            self.config.max_state_dim, self.vlm_with_expert.config.text_config.hidden_size
        )
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.vlm_with_expert.expert_hidden_size)
        self.action_out_proj = nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim)

        self.action_time_mlp_in = nn.Linear(
            self.vlm_with_expert.expert_hidden_size * 2, self.vlm_with_expert.expert_hidden_size
        )
        self.action_time_mlp_out = nn.Linear(
            self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size
        )
        self.skill_embedding = None
        self.transition_head = None
        self.phase_embedding = None
        self.phase_head = None
        self.atomic_previous_skill_embedding = None
        self.atomic_classifier = None
        if self._skill_linking_enabled():
            self.skill_embedding = nn.Embedding(
                self.config.skill_linking_num_skills + 1, self.vlm_with_expert.expert_hidden_size
            )
            nn.init.zeros_(self.skill_embedding.weight)
            self.transition_head = SkillTransitionHead(
                self.vlm_with_expert.config.text_config.hidden_size,
                self.vlm_with_expert.expert_hidden_size,
                self.config.skill_linking_num_skills + 2,
            )
        elif self._phase_masking_enabled():
            self.phase_embedding = nn.Embedding(2, self.vlm_with_expert.expert_hidden_size)
            nn.init.zeros_(self.phase_embedding.weight)
            self.phase_head = nn.Linear(self.vlm_with_expert.expert_hidden_size, 2)
        if self._atomic_classifier_enabled():
            self.atomic_previous_skill_embedding = nn.Embedding(
                len(ATOMIC_SKILLS) + 1, self.vlm_with_expert.expert_hidden_size
            )
            self.atomic_classifier = AtomicSkillClassifier(
                self.vlm_with_expert.config.text_config.hidden_size,
                self.vlm_with_expert.expert_hidden_size,
                self.config.max_state_dim,
            )

        self.set_requires_grad()
        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )

        self.add_image_special_tokens = self.config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = self.config.prefix_length
        self.rtc_processor = rtc_processor

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _skill_linking_enabled(self) -> bool:
        return getattr(self.config, "skill_linking_enabled", False)

    def _phase_masking_enabled(self) -> bool:
        return getattr(self.config, "phase_camera_masking_enabled", False)

    def _atomic_classifier_enabled(self) -> bool:
        return getattr(self.config, "atomic_classifier_enabled", False)

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj
        if self._atomic_classifier_enabled():
            for parameter in self.parameters():
                parameter.requires_grad = False
            for module in (self.atomic_previous_skill_embedding, self.atomic_classifier):
                for parameter in module.parameters():
                    parameter.requires_grad = True

    def sample_noise(self, shape, device):
        return sample_noise(shape, device)

    def sample_time(self, bsize, device):
        return sample_time_beta(bsize, device, alpha=1.5, beta=1.0, scale=0.999, offset=0.001)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, state: torch.Tensor = None
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[tuple[int, int], ...],
        tuple[int, int],
    ]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for SmolVLM transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []
        visual_token_spans = []
        for _img_idx, (
            img,
            img_mask,
        ) in enumerate(zip(images, img_masks, strict=False)):
            if self.add_image_special_tokens:
                image_start_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
                )
                att_masks += [0] * (image_start_mask.shape[-1])
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)

            img_emb = self.vlm_with_expert.embed_image(img)
            img_emb = img_emb

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            visual_start = sum(emb.shape[1] for emb in embs)
            embs.append(img_emb)
            visual_token_spans.append((visual_start, visual_start + num_img_embs))
            pad_masks.append(img_mask)

            att_masks += [0] * (num_img_embs)
            if self.add_image_special_tokens:
                image_end_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device
                )
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * (image_end_mask.shape[1])
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        task_token_span = (
            sum(emb.shape[1] for emb in embs),
            sum(emb.shape[1] for emb in embs) + lang_emb.shape[1],
        )
        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        device = state_emb.device

        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1] * (states_seq_len)
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        att_masks = att_masks.expand(bsize, -1)

        return embs, pad_masks, att_masks, tuple(visual_token_spans), task_token_span

    def embed_suffix(
        self,
        noisy_actions,
        timestep,
        current_skill: Tensor | None = None,
        current_phase: Tensor | None = None,
    ):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)
        if self._skill_linking_enabled():
            if current_skill is None:
                raise ValueError("Skill linking requires current_skill for suffix embedding.")
            skill_emb = self.skill_embedding(current_skill.long().to(device=device))[:, None, :]
            action_time_emb = action_time_emb + skill_emb.to(dtype=action_time_emb.dtype).expand_as(
                action_time_emb
            )
        elif self._phase_masking_enabled():
            if current_phase is None:
                raise ValueError("Phase-aware masking requires current_phase for suffix embedding.")
            phase_emb = self.phase_embedding(current_phase.long().to(device=device))[:, None, :]
            action_time_emb = action_time_emb + phase_emb.to(dtype=action_time_emb.dtype).expand_as(
                action_time_emb
            )

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] * self.config.chunk_size
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def _transition_logits_from_prefix(
        self,
        prefix_out: Tensor,
        prefix_pad_masks: Tensor,
        visual_token_spans: tuple[tuple[int, int], ...],
        task_token_span: tuple[int, int],
        current_skill: Tensor,
    ) -> Tensor:
        visual_tokens = torch.cat([prefix_out[:, start:end] for start, end in visual_token_spans], dim=1)
        visual_mask = torch.cat([prefix_pad_masks[:, start:end] for start, end in visual_token_spans], dim=1)
        task_start, task_end = task_token_span
        return self.transition_head(
            visual_tokens,
            visual_mask,
            prefix_out[:, task_start:task_end],
            prefix_pad_masks[:, task_start:task_end],
            self.skill_embedding(current_skill.long().to(device=prefix_out.device)),
        )

    def _phase_logits_from_suffix(self, suffix_out: Tensor) -> Tensor:
        horizon = min(self.config.n_action_steps, suffix_out.shape[1])
        pooled = suffix_out[:, :horizon].to(dtype=self.phase_head.weight.dtype).mean(dim=1)
        return self.phase_head(pooled).float()

    def classify_atomic_skill(
        self, images, img_masks, lang_tokens, lang_masks, state, previous_skill: Tensor
    ) -> Tensor:
        (
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            visual_token_spans,
            task_token_span,
        ) = self.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
        prefix_outputs, _ = self.vlm_with_expert.forward(
            attention_mask=make_att_2d_masks(prefix_pad_masks, prefix_att_masks),
            position_ids=torch.cumsum(prefix_pad_masks, dim=1) - 1,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
        )
        prefix_out = prefix_outputs[0]
        visual_tokens = torch.cat([prefix_out[:, start:end] for start, end in visual_token_spans], dim=1)
        visual_mask = torch.cat([prefix_pad_masks[:, start:end] for start, end in visual_token_spans], dim=1)
        task_start, task_end = task_token_span
        return self.atomic_classifier(
            visual_tokens,
            visual_mask,
            prefix_out[:, task_start:task_end],
            prefix_pad_masks[:, task_start:task_end],
            state,
            self.atomic_previous_skill_embedding(previous_skill.long().to(device=state.device)),
        )

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        noise=None,
        time=None,
        current_skill=None,
        current_phase=None,
        atomic_skill_id=None,
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        (
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            visual_token_spans,
            task_token_span,
        ) = self.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            x_t, time, current_skill=current_skill, current_phase=current_phase
        )

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (prefix_out, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            visual_token_spans=visual_token_spans,
            atomic_skill_id=atomic_skill_id,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        if self._skill_linking_enabled():
            return losses, self._transition_logits_from_prefix(
                prefix_out,
                prefix_pad_masks,
                visual_token_spans,
                task_token_span,
                current_skill,
            )
        if self._phase_masking_enabled():
            return losses, self._phase_logits_from_suffix(suffix_out)
        return losses

    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        current_skill: Tensor | None = None,
        current_phase: Tensor | None = None,
        atomic_skill_id: Tensor | None = None,
        task_descriptions: list[str] | tuple[str, ...] | str | None = None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        (
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            visual_token_spans,
            task_token_span,
        ) = self.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        # Compute image and language key value cache
        prefix_outputs, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
        )
        transition_logits = None
        if self._skill_linking_enabled():
            transition_logits = self._transition_logits_from_prefix(
                prefix_outputs[0],
                prefix_pad_masks,
                visual_token_spans,
                task_token_span,
                current_skill,
            )
        num_steps = self.config.num_steps
        latest_auxiliary_logits = None
        self.vlm_with_expert.start_focus_token_diagnostics_call(images)
        self.vlm_with_expert.start_attention_map_call(images, visual_token_spans, task_descriptions)
        try:

            def denoise_with_auxiliary(input_x_t, current_timestep):
                nonlocal latest_auxiliary_logits
                denoise_out = self.denoise_step(
                    x_t=input_x_t,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    timestep=current_timestep,
                    visual_token_spans=visual_token_spans,
                    current_skill=current_skill,
                    current_phase=current_phase,
                    atomic_skill_id=atomic_skill_id,
                )
                if self._phase_masking_enabled():
                    velocity, latest_auxiliary_logits = denoise_out
                    return velocity
                return denoise_out

            actions = euler_integrate(
                denoise_with_auxiliary,
                noise,
                num_steps,
                rtc_processor=self.rtc_processor,
                rtc_enabled=self._rtc_enabled(),
                inference_delay=kwargs.get("inference_delay"),
                prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
                execution_horizon=kwargs.get("execution_horizon"),
            )
            if self._skill_linking_enabled():
                return actions, transition_logits
            if not self._phase_masking_enabled():
                return actions
            if latest_auxiliary_logits is None:
                raise RuntimeError("Expected auxiliary logits from the final denoise step.")
            return actions, latest_auxiliary_logits
        finally:
            self.vlm_with_expert.end_attention_map_call()
            self.vlm_with_expert.end_focus_token_diagnostics_call()

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        visual_token_spans,
        current_skill: Tensor | None = None,
        current_phase: Tensor | None = None,
        atomic_skill_id: Tensor | None = None,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        if (
            self.vlm_with_expert.focus_token_diagnostics_path is not None
            or getattr(self.vlm_with_expert, "attention_map_collector", None) is not None
        ):
            timestep_value = float(timestep[0].item())
            denoising_step = round((1.0 - timestep_value) * self.config.num_steps)
            if self.vlm_with_expert.focus_token_diagnostics_path is not None:
                self.vlm_with_expert.focus_token_diagnostics_context.update(
                    denoising_step=denoising_step,
                    denoising_timestep=timestep_value,
                )
            self.vlm_with_expert.set_attention_map_flow_step(denoising_step)
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            x_t, timestep, current_skill=current_skill, current_phase=current_phase
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            visual_token_spans=visual_token_spans,
            atomic_skill_id=atomic_skill_id,
        )
        if past_key_values is not None:
            # Self-attention layers append suffix K/V in place; restore the prefix for the next step.
            past_key_values.crop(prefix_len)
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        if self._phase_masking_enabled():
            return v_t, self._phase_logits_from_suffix(suffix_out)
        return v_t
