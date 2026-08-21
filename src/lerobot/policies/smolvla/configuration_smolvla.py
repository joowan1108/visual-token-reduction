# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import math
from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import OBS_IMAGES

from ..rtc.configuration_rtc import RTCConfig


@PreTrainedConfig.register_subclass("smolvla")
@dataclass
class SmolVLAConfig(PreTrainedConfig):
    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Shorter state and action vectors will be padded
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (512, 512)

    # Add empty images. Used by smolvla_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    empty_cameras: int = 0

    # Converts the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi_aloha: bool = False

    # Converts joint dimensions to relative values with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions_aloha: bool = False

    # Tokenizer
    tokenizer_max_length: int = 48

    # Decoding
    num_steps: int = 10

    # Attention utils
    use_cache: bool = True

    # Finetuning settings
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True

    # AtomicVLA-style data handling is shared by the dense and SG-MoE conditions.
    atomic_data_enabled: bool = False
    atomic_sgmoe_enabled: bool = False
    atomic_planner_enabled: bool = False
    atomic_classifier_enabled: bool = False
    atomic_anchor_stride: int = 1
    atomic_subtask_to_skill: list[int] | None = None
    atomic_subtask_to_skill_path: str | None = None

    # Implicit action reasoning with a FAST next-token auxiliary objective. Opt-in only.
    implicit_fast_ki_enabled: bool = False
    implicit_iar_layers: list[int] = field(default_factory=lambda: [-4, -3, -2, -1])
    implicit_iar_num_queries: int = 4
    implicit_fast_loss_weight: float = 0.1
    implicit_transition_loss_weight: float = 0.1
    implicit_transition_focal_gamma: float = 2.0
    implicit_fast_max_action_tokens: int = 256
    implicit_fast_skip_tokens: int = 128
    implicit_fast_action_tokenizer_name: str = "lerobot/fast-action-tokenizer"

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"  # Select the VLM backbone.
    load_vlm_weights: bool = False  # Set to False in case of training the expert from scratch. True when init from pretrained SmolVLA weights

    add_image_special_tokens: bool = False  # Whether to use special image tokens around image features.

    attention_mode: str = "cross_attn"

    prefix_length: int = -1

    pad_language_to: str = "longest"  # "max_length"

    num_expert_layers: int = -1  # Less or equal to 0 is the default where the action expert has the same number of layers of VLM. Otherwise the expert have less layers.
    num_vlm_layers: int = 16  # Number of layers used in the VLM (first num_vlm_layers layers)
    self_attn_every_n_layers: int = 2  # Interleave SA layers each self_attn_every_n_layers
    expert_width_multiplier: float = 0.75  # The action expert hidden size (wrt to the VLM)

    # Late action-aware visual token selection. A ratio of 1.0 preserves the dense baseline.
    focus_token_keep_ratio: float = 1.0
    focus_token_start_layer: int = 8
    focus_token_diagnostics_path: str | None = None
    focus_cascaded_attention: bool = False
    focus_channel_gate: bool = False
    attention_map: bool = False
    attention_map_output_dir: str | None = None
    attention_map_layers: list[int] = field(default_factory=lambda: [-4, -3, -2, -1])
    attention_map_flow_steps: list[int] = field(default_factory=lambda: [-1])

    # Optional semantic skill linking for LIBERO-10. Disabled by default.
    skill_linking_enabled: bool = False
    skill_linking_sampler_enabled: bool = False
    skill_linking_num_skills: int = 16
    skill_transition_loss_weight: float = 0.1
    skill_transition_class_weights: list[float] | None = None

    # Long-VLA-style phase-aware camera masking. Labels are moving=-1 and interaction=+1.
    phase_camera_masking_enabled: bool = False
    phase_loss_weight: float = 0.1
    phase_weak_split_ratio: float | None = 0.75
    phase_static_camera_key: str = f"{OBS_IMAGES}.image"
    phase_wrist_camera_key: str = f"{OBS_IMAGES}.image2"

    min_period: float = 4e-3  # sensitivity range for the timestep used in sine-cosine positional encoding
    max_period: float = 4.0

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by smolvla for aloha real models. It is not ported yet in LeRobot."
            )
        if not 0 < self.focus_token_keep_ratio <= 1:
            raise ValueError("`focus_token_keep_ratio` must be in (0, 1].")
        if not 0 <= self.focus_token_start_layer < self.num_vlm_layers:
            raise ValueError("`focus_token_start_layer` must index a VLM layer.")
        if self.focus_channel_gate and not self.focus_cascaded_attention:
            raise ValueError("`focus_channel_gate=True` requires `focus_cascaded_attention=True`.")
        if self.attention_map:
            if not self.attention_map_output_dir:
                raise ValueError("`attention_map_output_dir` is required when `attention_map=True`.")
            if not self.attention_map_layers or not self.attention_map_flow_steps:
                raise ValueError("Attention-map layers and flow steps must not be empty.")
            if any(
                not -self.num_vlm_layers <= layer < self.num_vlm_layers for layer in self.attention_map_layers
            ):
                raise ValueError("`attention_map_layers` contains an out-of-range layer index.")
            if self.compile_model:
                raise ValueError("Attention-map capture requires `compile_model=False`.")
        if self.focus_token_keep_ratio < 1 or self.focus_cascaded_attention:
            if (
                self.num_vlm_layers != 16
                or self.num_expert_layers not in (-1, 16)
                or self.attention_mode != "cross_attn"
                or self.self_attn_every_n_layers != 2
            ):
                raise ValueError(
                    "Focus-token experiments require 16 VLM/expert layers, cross_attn, and "
                    "self_attn_every_n_layers=2."
                )
            if self.compile_model:
                raise ValueError("Focus-token experiments require `compile_model=False`.")
        if self.skill_linking_enabled:
            if self.skill_linking_num_skills <= 0:
                raise ValueError("`skill_linking_num_skills` must be positive.")
            if not self.train_expert_only or not self.freeze_vision_encoder:
                raise ValueError("Skill linking requires a frozen VLM and vision encoder.")
            if self.n_action_steps <= 0:
                raise ValueError("Skill linking requires `n_action_steps > 0`.")
            if self.compile_model:
                raise ValueError("Skill linking requires `compile_model=False`.")
            if self.rtc_config is not None and self.rtc_config.enabled:
                raise ValueError("Skill linking does not support RTC.")
            weights = self.skill_transition_class_weights
            if weights is not None:
                if len(weights) != self.skill_linking_num_skills + 2:
                    raise ValueError("`skill_transition_class_weights` must contain num_skills + 2 values.")
                if weights[0] != 0 or any(not math.isfinite(weight) or weight <= 0 for weight in weights[1:]):
                    raise ValueError(
                        "Transition class weight 0 must be zero; every other weight must be positive and finite."
                    )
        if self.skill_linking_sampler_enabled and self.n_action_steps <= 0:
            raise ValueError("Skill-linking sampling requires `n_action_steps > 0`.")
        if (
            isinstance(self.atomic_anchor_stride, bool)
            or not isinstance(self.atomic_anchor_stride, int)
            or self.atomic_anchor_stride < 1
        ):
            raise ValueError("`atomic_anchor_stride` must be an integer >= 1.")
        if self.atomic_data_enabled:
            if self.chunk_size != 10 or not 0 < self.n_action_steps <= 10:
                raise ValueError("Atomic experiments require chunk_size=10 and 1 <= n_action_steps <= 10.")
            if self.atomic_subtask_to_skill is None and self.atomic_subtask_to_skill_path is None:
                raise ValueError(
                    "`atomic_subtask_to_skill` or `atomic_subtask_to_skill_path` is required when atomic data handling is enabled."
                )
            if self.atomic_subtask_to_skill is not None and set(self.atomic_subtask_to_skill) != set(
                range(6)
            ):
                raise ValueError("`atomic_subtask_to_skill` must cover exactly the six skill IDs 0..5.")
            if self.skill_linking_enabled or self.skill_linking_sampler_enabled:
                raise ValueError("Atomic SG-MoE and semantic skill linking are separate experiments.")
            if self.phase_camera_masking_enabled:
                raise ValueError("Atomic SG-MoE and phase-aware masking are separate experiments.")
        if self.atomic_sgmoe_enabled:
            if not self.atomic_data_enabled:
                raise ValueError("Atomic SG-MoE requires `atomic_data_enabled=True`.")
            if not self.train_expert_only:
                raise ValueError("Atomic SG-MoE requires the VLM to be frozen with `train_expert_only=True`.")
            if self.compile_model:
                raise ValueError("Atomic SG-MoE currently requires `compile_model=False`.")
        if self.atomic_planner_enabled:
            if not self.atomic_sgmoe_enabled:
                raise ValueError("The frozen atomic planner requires `atomic_sgmoe_enabled=True`.")
            if not self.train_expert_only:
                raise ValueError("The frozen atomic planner must share a frozen VLM.")
            if not self.freeze_vision_encoder:
                raise ValueError(
                    "The frozen atomic planner requires a frozen vision encoder "
                    "(`freeze_vision_encoder=True`)."
                )
            if not self.load_vlm_weights:
                raise ValueError("The frozen atomic planner requires `load_vlm_weights=True`.")
        if self.atomic_classifier_enabled:
            if not self.atomic_sgmoe_enabled:
                raise ValueError("The atomic classifier requires `atomic_sgmoe_enabled=True`.")
            if self.atomic_planner_enabled:
                raise ValueError("The atomic classifier and generative planner are mutually exclusive.")
            if not self.train_expert_only or not self.freeze_vision_encoder:
                raise ValueError("The atomic classifier requires a fully frozen action-policy backbone.")
        if self.implicit_fast_ki_enabled:
            if not self.atomic_sgmoe_enabled:
                raise ValueError("Implicit FAST-KI requires `atomic_sgmoe_enabled=True`.")
            if not self.train_expert_only or not self.freeze_vision_encoder:
                raise ValueError("Implicit FAST-KI requires a fully frozen VLM and vision encoder.")
            if not self.load_vlm_weights and self.pretrained_path is None:
                raise ValueError(
                    "Implicit FAST-KI requires `load_vlm_weights=True` or a pretrained policy checkpoint."
                )
            if not self.use_cache:
                raise ValueError("Implicit FAST-KI requires VLM KV caching.")
            if self.compile_model:
                raise ValueError("Implicit FAST-KI currently requires `compile_model=False`.")
            if not self.implicit_iar_layers:
                raise ValueError("`implicit_iar_layers` must select at least one VLM layer.")
            if any(
                not -self.num_vlm_layers <= layer < self.num_vlm_layers for layer in self.implicit_iar_layers
            ):
                raise ValueError("`implicit_iar_layers` contains an out-of-range VLM layer.")
            normalized_iar_layers = [layer % self.num_vlm_layers for layer in self.implicit_iar_layers]
            if len(set(normalized_iar_layers)) != len(normalized_iar_layers):
                raise ValueError("`implicit_iar_layers` must be unique after layer normalization.")
            if self.implicit_iar_num_queries <= 0:
                raise ValueError("`implicit_iar_num_queries` must be positive.")
            if not math.isfinite(self.implicit_fast_loss_weight) or self.implicit_fast_loss_weight <= 0:
                raise ValueError("`implicit_fast_loss_weight` must be positive and finite.")
            if (
                not math.isfinite(self.implicit_transition_loss_weight)
                or self.implicit_transition_loss_weight <= 0
            ):
                raise ValueError("`implicit_transition_loss_weight` must be positive and finite.")
            if (
                not math.isfinite(self.implicit_transition_focal_gamma)
                or self.implicit_transition_focal_gamma < 0
            ):
                raise ValueError("`implicit_transition_focal_gamma` must be finite and non-negative.")
            if self.implicit_fast_max_action_tokens < 2:
                raise ValueError("`implicit_fast_max_action_tokens` must be at least 2.")
            if self.atomic_classifier_enabled:
                raise ValueError("Implicit FAST-KI and classifier-only training are separate experiments.")
            if self.atomic_planner_enabled:
                raise ValueError(
                    "Implicit FAST-KI uses its dedicated transition head, not the atomic planner."
                )
            if not self.train_state_proj:
                raise ValueError("Implicit FAST-KI requires `train_state_proj=True`.")
        if self.phase_camera_masking_enabled:
            if self.skill_linking_enabled:
                raise ValueError("Phase-aware masking and semantic skill linking are separate experiments.")
            if self.focus_token_keep_ratio < 1 or self.focus_cascaded_attention:
                raise ValueError("Phase-aware masking must be evaluated without focus-token interventions.")
            if self.n_action_steps <= 0:
                raise ValueError("Phase-aware masking requires `n_action_steps > 0`.")
            if self.compile_model:
                raise ValueError("Phase-aware masking requires `compile_model=False`.")
            if self.rtc_config is not None and self.rtc_config.enabled:
                raise ValueError("Phase-aware masking does not support RTC.")
            if not math.isfinite(self.phase_loss_weight) or self.phase_loss_weight <= 0:
                raise ValueError("`phase_loss_weight` must be positive and finite.")
            if self.phase_weak_split_ratio is not None and not 0 < self.phase_weak_split_ratio < 1:
                raise ValueError("`phase_weak_split_ratio` must be in (0, 1) or None.")
            if self.phase_static_camera_key == self.phase_wrist_camera_key:
                raise ValueError("Static and wrist camera keys must be different.")

    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera
        if self.phase_camera_masking_enabled:
            expected = {self.phase_static_camera_key, self.phase_wrist_camera_key}
            actual = set(self.image_features)
            if actual != expected:
                raise ValueError(
                    "Phase-aware masking requires exactly the configured static and wrist cameras; "
                    f"expected {sorted(expected)}, got {sorted(actual)}."
                )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def subtask_delta_indices(self) -> list[int] | None:
        if self.atomic_data_enabled:
            if self.implicit_fast_ki_enabled:
                return [-2 * self.n_action_steps, -self.n_action_steps, *range(self.chunk_size)]
            return ([-1] if self.atomic_classifier_enabled else []) + list(range(self.chunk_size))
        return list(range(self.n_action_steps + 1)) if self.skill_linking_enabled else None

    @property
    def phase_delta_indices(self) -> list[int] | None:
        return [0, self.n_action_steps] if self.phase_camera_masking_enabled else None
