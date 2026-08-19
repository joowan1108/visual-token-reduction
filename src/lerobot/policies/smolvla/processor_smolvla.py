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

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor import (
    ActionTokenizerProcessorStep,
    NewLineTaskProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    TokenizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.utils.constants import ACTION_TOKEN_MASK, ACTION_TOKENS

from .configuration_smolvla import SmolVLAConfig


@dataclass
@ProcessorStepRegistry.register(name="smolvla_implicit_fast_action_tokenizer_processor")
class SmolVLAImplicitFastActionTokenizerProcessorStep(ActionTokenizerProcessorStep):
    atomic_subtask_to_skill: list[int] | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        complementary = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        labels = complementary.get("subtask_index")
        labels_is_pad = complementary.get("subtask_index_is_pad")
        if labels is None or labels_is_pad is None or self.atomic_subtask_to_skill is None:
            raise KeyError("Implicit FAST-KI requires atomic subtask labels and the frozen mapping.")
        if labels.ndim != 2 or labels.shape != labels_is_pad.shape or labels.shape[1] != action.shape[1] + 2:
            raise ValueError(
                "Implicit atomic labels must contain two history anchors plus the action horizon."
            )
        labels = labels[:, 2:]
        labels_is_pad = labels_is_pad[:, 2:]

        mapping = torch.tensor(self.atomic_subtask_to_skill, device=labels.device)
        valid = ~labels_is_pad.bool()
        if ((labels[valid] < 0) | (labels[valid] >= mapping.numel())).any():
            raise ValueError("Atomic subtask index is outside the frozen mapping vocabulary.")
        skills = mapping[labels.long().clamp(0, mapping.numel() - 1)]
        action_is_pad = labels_is_pad.bool() | (skills != skills[:, :1])
        existing = complementary.get("action_is_pad")
        if existing is not None:
            action_is_pad |= existing.bool()

        tokens, token_mask = self._tokenize_action(action, action_is_pad)
        complementary["action_is_pad"] = action_is_pad
        complementary[ACTION_TOKENS] = tokens
        complementary[ACTION_TOKEN_MASK] = token_mask
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return transition

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "fast_skip_tokens": self.fast_skip_tokens,
            "paligemma_tokenizer_name": self.paligemma_tokenizer_name,
            "atomic_subtask_to_skill": self.atomic_subtask_to_skill,
        }


def make_smolvla_pre_post_processors(
    config: SmolVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the SmolVLA policy.

    The pre-processing pipeline prepares input data for the model by:
    1.  Renaming features to match pretrained configurations.
    2.  Normalizing input and output features based on dataset statistics.
    3.  Adding a batch dimension.
    4.  Ensuring the language task description ends with a newline character.
    5.  Tokenizing the language task description.
    6.  Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1.  Moving data to the CPU.
    2.  Unnormalizing the output actions to their original scale.

    Args:
        config: The configuration object for the SmolVLA policy.
        dataset_stats: A dictionary of statistics for normalization.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    steps = make_default_policy_processor_steps(config, dataset_stats)

    input_steps = [
        steps.rename_observations,  # To mimic the same processor as pretrained one
        steps.add_batch_dim,
        NewLineTaskProcessorStep(),
        TokenizerProcessorStep(
            tokenizer_name=config.vlm_model_name,
            padding=config.pad_language_to,
            padding_side="right",
            max_length=config.tokenizer_max_length,
        ),
    ]
    if config.implicit_fast_ki_enabled:
        if config.atomic_subtask_to_skill is None:
            raise ValueError("Implicit FAST-KI requires the resolved atomic subtask mapping.")
        input_steps.extend(
            [
                steps.normalize,
                SmolVLAImplicitFastActionTokenizerProcessorStep(
                    action_tokenizer_name=config.implicit_fast_action_tokenizer_name,
                    max_action_tokens=config.implicit_fast_max_action_tokens,
                    fast_skip_tokens=config.implicit_fast_skip_tokens,
                    paligemma_tokenizer_name=config.vlm_model_name,
                    atomic_subtask_to_skill=config.atomic_subtask_to_skill,
                ),
                steps.to_device,
            ]
        )
    else:
        input_steps.extend([steps.to_device, steps.normalize])
    output_steps = [
        steps.unnormalize,
        steps.to_cpu,
    ]
    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
