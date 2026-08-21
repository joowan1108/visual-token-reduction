import inspect
import math
from collections import deque
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import lerobot.policies.smolvla.smolvlm_with_expert as smolvlm_with_expert
from lerobot.datasets.sampler import AtomicSkillSampler
from lerobot.lerobot_types import TransitionKey
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import (
    ATOMIC_SKILLS,
    AtomicPlannerEpisodeFailure,
    ImplicitAtomicTransitionHead,
    SmolVLAPolicy,
    VLAFlowMatching,
    _implicit_transition_focal_loss,
    atomic_classifier_event_counts,
    parse_atomic_planner_output,
)
from lerobot.policies.smolvla.processor_smolvla import (
    SmolVLAImplicitFastActionTokenizerProcessorStep,
)
from lerobot.policies.smolvla.smolvlm_with_expert import (
    AtomicSkillFFN,
    AtomicSkillRouter,
    ImplicitActionReasoner,
    SmolVLMWithExpertModel,
)
from lerobot.utils.constants import ACTION, ACTION_TOKEN_MASK, ACTION_TOKENS, OBS_STATE


def test_atomic_config_keeps_dense_defaults_and_freezes_temporal_contract():
    assert SmolVLAConfig().atomic_sgmoe_enabled is False
    assert SmolVLAConfig().atomic_anchor_stride == 1
    for invalid_stride in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="atomic_anchor_stride"):
            SmolVLAConfig(atomic_anchor_stride=invalid_stride)
    config = SmolVLAConfig(
        chunk_size=10,
        n_action_steps=5,
        atomic_data_enabled=True,
        atomic_sgmoe_enabled=True,
        atomic_subtask_to_skill=list(range(6)),
    )
    assert config.subtask_delta_indices == list(range(10))
    thinned = SmolVLAConfig(
        chunk_size=10,
        n_action_steps=10,
        atomic_data_enabled=True,
        atomic_sgmoe_enabled=True,
        atomic_anchor_stride=5,
        atomic_subtask_to_skill=list(range(6)),
    )
    assert thinned.subtask_delta_indices == list(range(10))
    with pytest.raises(ValueError, match="load_vlm_weights=True"):
        SmolVLAConfig(
            chunk_size=10,
            n_action_steps=5,
            atomic_data_enabled=True,
            atomic_sgmoe_enabled=True,
            atomic_planner_enabled=True,
            atomic_subtask_to_skill=list(range(6)),
        )
    planner = SmolVLAConfig(
        chunk_size=10,
        n_action_steps=5,
        atomic_data_enabled=True,
        atomic_sgmoe_enabled=True,
        atomic_planner_enabled=True,
        atomic_subtask_to_skill=list(range(6)),
        load_vlm_weights=True,
    )
    assert planner.atomic_planner_enabled

    classifier = SmolVLAConfig(
        chunk_size=10,
        n_action_steps=5,
        atomic_data_enabled=True,
        atomic_sgmoe_enabled=True,
        atomic_classifier_enabled=True,
        atomic_subtask_to_skill=list(range(6)),
    )
    assert classifier.subtask_delta_indices == [-1, *range(10)]

    assert replace(classifier, n_action_steps=10).n_action_steps == 10

    with pytest.raises(ValueError, match="frozen vision encoder"):
        SmolVLAConfig(
            chunk_size=10,
            n_action_steps=5,
            atomic_data_enabled=True,
            atomic_sgmoe_enabled=True,
            atomic_planner_enabled=True,
            atomic_subtask_to_skill=list(range(6)),
            load_vlm_weights=True,
            freeze_vision_encoder=False,
        )


def test_atomic_router_matches_upstream_initialization_and_is_global_per_sample():
    router = AtomicSkillRouter(hidden_size=8)
    expected_scale = math.log(5) / 55
    torch.testing.assert_close(router.route.weight[:, :6], torch.eye(6) * expected_scale, rtol=0, atol=0)
    torch.testing.assert_close(router.skill_embeddings[:, :6], torch.diag(torch.linspace(10.0, 100.0, 6)))

    selected, weight = router(torch.arange(6))
    assert selected.tolist() == list(range(6))
    assert selected.shape == weight.shape == (6,)


def test_atomic_ffn_starts_dense_equivalent_and_only_runs_selected_experts():
    torch.manual_seed(0)
    dense = nn.Sequential(nn.Linear(4, 7), nn.SiLU(), nn.Linear(7, 4))
    atomic = AtomicSkillFFN(dense)
    router = AtomicSkillRouter(hidden_size=8)
    route = router(torch.tensor([0, 2]))
    x = torch.randn(2, 5, 4)

    torch.testing.assert_close(atomic(x, route), dense(x))
    atomic(x, route).sum().backward()
    assert atomic.shared_expert[0].weight.grad is not None
    assert atomic.skill_experts[0][0].weight.grad is not None
    assert atomic.skill_experts[2][0].weight.grad is not None
    assert all(atomic.skill_experts[index][0].weight.grad is None for index in (1, 3, 4, 5))
    assert router.route.weight.grad is not None


def test_implicit_fast_ki_layer_independence_gradient_isolation_and_no_leakage():
    assert SmolVLAConfig().implicit_fast_ki_enabled is False
    implicit_config = {
        "chunk_size": 10,
        "n_action_steps": 5,
        "atomic_data_enabled": True,
        "atomic_sgmoe_enabled": True,
        "implicit_fast_ki_enabled": True,
        "atomic_subtask_to_skill": list(range(6)),
    }
    with pytest.raises(ValueError, match="pretrained policy checkpoint"):
        SmolVLAConfig(**implicit_config)
    with pytest.raises(ValueError, match="pretrained policy checkpoint"):
        VLAFlowMatching(
            SimpleNamespace(
                implicit_fast_ki_enabled=True,
                load_vlm_weights=False,
                pretrained_path=None,
            )
        )
    with pytest.raises(ValueError, match="unique after layer normalization"):
        SmolVLAConfig(
            **implicit_config,
            pretrained_path=Path("local-checkpoint"),
            implicit_iar_layers=[-1, 15],
        )
    with pytest.raises(ValueError, match="train_state_proj=True"):
        SmolVLAConfig(
            **implicit_config,
            pretrained_path=Path("local-checkpoint"),
            train_state_proj=False,
        )
    config = SmolVLAConfig(
        **implicit_config,
        pretrained_path=Path("local-checkpoint"),
    )
    assert config.implicit_iar_layers == [-4, -3, -2, -1]
    assert config.train_state_proj is True
    reasoner = ImplicitActionReasoner([0, 1], kv_size=6, hidden_size=4, num_queries=3)
    assert reasoner.queries[0] is not reasoner.queries[1]
    assert reasoner.query_projections[0] is not reasoner.query_projections[1]
    assert reasoner.key_projections[0] is not reasoner.key_projections[1]
    assert reasoner.value_projections[0] is not reasoner.value_projections[1]

    vlm_weight = nn.Parameter(torch.tensor(1.0))
    layers = []
    for _ in range(2):
        layers.append(
            SimpleNamespace(
                keys=vlm_weight * torch.randn(2, 2, 5, 3),
                values=vlm_weight * torch.randn(2, 2, 5, 3),
            )
        )
    cache = SimpleNamespace(layers=layers)
    prefix_mask = torch.tensor([[True, True, True, False, False], [True] * 5])

    layer_one_before = reasoner.project_layer(1, layers[1].keys, layers[1].values, prefix_mask)
    with torch.no_grad():
        reasoner.key_projections[0].weight.add_(1)
    layer_one_after = reasoner.project_layer(1, layers[1].keys, layers[1].values, prefix_mask)
    torch.testing.assert_close(layer_one_before, layer_one_after)

    context = reasoner(cache, prefix_mask)
    assert context.shape == (2, 3, 4)
    action_targets = torch.randn(2, 10, 7)
    unchanged_context = reasoner(cache, prefix_mask)
    action_targets.add_(1000)
    torch.testing.assert_close(reasoner(cache, prefix_mask), unchanged_context)

    class TinyVLMWithExpert(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(16, 4)
            self.lm_head = nn.Linear(4, 16)
            self.vlm = SimpleNamespace(lm_head=self.lm_head)
            self.sgmoe = AtomicSkillFFN(nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 4)))
            self.atomic_router = AtomicSkillRouter(6)
            self.embedding.requires_grad_(False)
            self.lm_head.requires_grad_(False)

        def embed_image(self, image):
            return image

        def embed_language_tokens(self, tokens):
            return self.embedding(tokens)

        def get_vlm_model(self):
            return SimpleNamespace(text_model=SimpleNamespace(get_input_embeddings=lambda: self.embedding))

        def forward(self, inputs_embeds, atomic_skill_id=None, **kwargs):
            implicit, suffix = inputs_embeds
            if suffix is None:
                return [implicit + implicit.mean(dim=1, keepdim=True), None], None
            route = (atomic_skill_id, torch.full_like(atomic_skill_id, 0.5, dtype=suffix.dtype))
            mixed = suffix + implicit.mean(dim=1, keepdim=True)
            return [implicit, self.sgmoe(mixed, route)], None

    flow_model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(flow_model)
    flow_model.config = SimpleNamespace(
        implicit_fast_ki_enabled=True,
        train_state_proj=True,
        atomic_classifier_enabled=False,
        chunk_size=5,
    )
    flow_model.state_proj = nn.Linear(4, 4)
    flow_model.fast_context_proj = nn.Linear(4, 4)
    flow_model.implicit_transition_head = ImplicitAtomicTransitionHead(4, 6)
    flow_model.vlm_with_expert = TinyVLMWithExpert()
    flow_model.implicit_action_reasoner = reasoner
    flow_model.add_image_special_tokens = False
    flow_model.prefix_length = -1
    flow_model.set_requires_grad()
    assert all(parameter.requires_grad for parameter in flow_model.state_proj.parameters())
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    nn.Module.__init__(policy)
    policy.config = config
    policy.model = flow_model
    optim_param_ids = {id(parameter) for parameter in policy.get_optim_params()}
    assert all(id(parameter) in optim_param_ids for parameter in flow_model.state_proj.parameters())

    images = [torch.randn(2, 2, 4)]
    image_masks = [torch.ones(2, dtype=torch.bool)]
    language_tokens = torch.tensor([[1, 2], [3, 4]])
    language_masks = torch.ones(2, 2, dtype=torch.bool)
    prefix_without_state = flow_model.embed_prefix(
        images, image_masks, language_tokens, language_masks, state=None
    )[0]
    changed_state = torch.randn(2, 4)
    iar_before_state_change = flow_model._implicit_context(cache, prefix_mask)
    changed_state.add_(1000)
    torch.testing.assert_close(flow_model._implicit_context(cache, prefix_mask), iar_before_state_change)
    torch.testing.assert_close(
        flow_model.embed_prefix(images, image_masks, language_tokens, language_masks, state=None)[0],
        prefix_without_state,
    )
    assert (
        flow_model.embed_prefix(images, image_masks, language_tokens, language_masks, state=changed_state)[
            0
        ].shape[1]
        == prefix_without_state.shape[1] + 1
    )
    assert "state=None if self._implicit_fast_ki_enabled() else state" in inspect.getsource(
        VLAFlowMatching.forward
    )
    assert "state=None if self._implicit_fast_ki_enabled() else state" in inspect.getsource(
        VLAFlowMatching.sample_actions
    )

    state = torch.randn(2, 4)
    fused_context = flow_model._fused_implicit_context(context, state)
    assert fused_context.shape == (2, 4, 4)
    fast_tokens = torch.tensor([[1, 2, 3], [2, 3, 4]])
    fast_masks = torch.ones_like(fast_tokens, dtype=torch.bool)
    flow_model._fast_losses(fused_context, fast_tokens, fast_masks).mean().backward()
    assert any(parameter.grad is not None for parameter in reasoner.parameters())
    assert any(parameter.grad is not None for parameter in flow_model.fast_context_proj.parameters())
    assert any(parameter.grad is not None for parameter in flow_model.state_proj.parameters())
    assert all(parameter.grad is None for parameter in flow_model.vlm_with_expert.sgmoe.parameters())
    assert flow_model.vlm_with_expert.embedding.weight.grad is None
    assert flow_model.vlm_with_expert.lm_head.weight.grad is None
    assert vlm_weight.grad is None

    for module in (
        reasoner,
        flow_model.fast_context_proj,
        flow_model.state_proj,
        flow_model.vlm_with_expert.sgmoe,
    ):
        for parameter in module.parameters():
            parameter.grad = None

    route = (torch.tensor([0, 1]), torch.tensor([0.5, 0.5]))
    suffix = torch.randn(2, 5, 4)
    flow_context = reasoner(cache, prefix_mask)
    fused_context = flow_model._fused_implicit_context(flow_context, state)
    flow_model._forward_implicit_action(
        fused_context,
        suffix,
        torch.ones(2, 5, dtype=torch.bool),
        torch.ones(2, 5, dtype=torch.bool),
        route[0],
    ).square().mean().backward()
    assert any(parameter.grad is not None for parameter in flow_model.vlm_with_expert.sgmoe.parameters())
    assert all(parameter.grad is None for parameter in reasoner.parameters())
    assert all(parameter.grad is None for parameter in flow_model.fast_context_proj.parameters())
    assert all(parameter.grad is None for parameter in flow_model.state_proj.parameters())
    assert flow_model.vlm_with_expert.embedding.weight.grad is None
    assert "_fast_losses" not in inspect.getsource(VLAFlowMatching.sample_actions)
    assert vlm_weight.grad is None

    for module in (
        reasoner,
        flow_model.fast_context_proj,
        flow_model.state_proj,
        flow_model.vlm_with_expert.sgmoe,
        flow_model.implicit_transition_head,
    ):
        for parameter in module.parameters():
            parameter.grad = None
    transition_context = flow_model._fused_implicit_context(reasoner(cache, prefix_mask), state)
    no_history = torch.zeros(2, 2, dtype=torch.bool)
    no_history_logits = flow_model._implicit_transition_logits(
        transition_context, torch.tensor([[0, 1], [2, 3]]), no_history
    )
    torch.testing.assert_close(
        no_history_logits,
        flow_model._implicit_transition_logits(
            transition_context, torch.tensor([[5, 4], [1, 0]]), no_history
        ),
    )
    assert no_history_logits.shape == (2, len(ATOMIC_SKILLS))
    with pytest.raises(ValueError, match=r"\[batch, 2\]"):
        flow_model._implicit_transition_logits(
            transition_context,
            torch.zeros(2, 3, dtype=torch.long),
            torch.zeros(2, 3, dtype=torch.bool),
        )
    history_ids = torch.tensor([[0, 1], [2, 3]])
    history_valid = torch.ones(2, 2, dtype=torch.bool)
    transition_logits = flow_model._implicit_transition_logits(transition_context, history_ids, history_valid)
    transition_target = torch.tensor([1, 4])
    _implicit_transition_focal_loss(
        torch.nn.functional.cross_entropy(transition_logits, transition_target, reduction="none"),
        gamma=config.implicit_transition_focal_gamma,
    ).mean().backward()
    assert any(parameter.grad is not None for parameter in flow_model.implicit_transition_head.parameters())
    assert all(parameter.grad is None for parameter in reasoner.parameters())
    assert all(parameter.grad is None for parameter in flow_model.fast_context_proj.parameters())
    assert all(parameter.grad is None for parameter in flow_model.state_proj.parameters())
    assert all(parameter.grad is None for parameter in flow_model.vlm_with_expert.sgmoe.parameters())
    assert flow_model.vlm_with_expert.atomic_router.skill_embeddings.requires_grad is False
    assert flow_model.vlm_with_expert.atomic_router.route.weight.grad is None


def test_implicit_fast_tokenizer_ignores_pick_place_pick_boundaries():
    tokenizer = SmolVLAImplicitFastActionTokenizerProcessorStep.__new__(
        SmolVLAImplicitFastActionTokenizerProcessorStep
    )
    tokenizer.atomic_subtask_to_skill = [0, 1]
    captured_masks = []

    def tokenize(actions, action_is_pad):
        captured_masks.append(action_is_pad.clone())
        return torch.ones(1, 3, dtype=torch.long), torch.ones(1, 3, dtype=torch.bool)

    tokenizer._tokenize_action = tokenize
    tokenized = tokenizer(
        {
            TransitionKey.OBSERVATION: {},
            TransitionKey.ACTION: torch.zeros(1, 4, 2),
            TransitionKey.REWARD: None,
            TransitionKey.DONE: None,
            TransitionKey.TRUNCATED: None,
            TransitionKey.INFO: None,
            TransitionKey.COMPLEMENTARY_DATA: {
                "subtask_index": torch.tensor([[0, 0, 0, 1, 0, 0]]),
                "subtask_index_is_pad": torch.tensor([[True, True, False, False, False, True]]),
                "action_is_pad": torch.tensor([[False, False, False, True]]),
            },
        }
    )
    assert captured_masks[0].tolist() == [[False, False, False, True]]
    assert not ((~captured_masks[0][:, 1:]) & captured_masks[0][:, :-1]).any()
    assert tokenized[TransitionKey.COMPLEMENTARY_DATA]["action_is_pad"].tolist() == [
        [False, False, False, True]
    ]
    assert ACTION_TOKENS in tokenized[TransitionKey.COMPLEMENTARY_DATA]
    assert ACTION_TOKEN_MASK in tokenized[TransitionKey.COMPLEMENTARY_DATA]


def test_atomic_action_padding_ignores_pick_place_pick_boundaries():
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    policy.config = SimpleNamespace(chunk_size=4, atomic_subtask_to_skill=list(range(6)))
    skill, action_is_pad = policy._atomic_batch_contract(
        {
            "subtask_index": torch.tensor([[0, 1, 0, 0]]),
            "subtask_index_is_pad": torch.tensor([[False, False, False, True]]),
        }
    )
    assert skill.tolist() == [0]
    assert action_is_pad.tolist() == [[False, False, False, True]]


def test_implicit_transition_history_anchors_are_episode_safe_and_reset_per_batch():
    config = SmolVLAConfig(
        chunk_size=10,
        n_action_steps=10,
        atomic_data_enabled=True,
        atomic_sgmoe_enabled=True,
        implicit_fast_ki_enabled=True,
        atomic_subtask_to_skill=list(range(6)),
        pretrained_path=Path("local-checkpoint"),
    )
    assert config.subtask_delta_indices == [-20, -10, *range(10)]
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    nn.Module.__init__(policy)
    policy.config = config
    labels = torch.tensor(
        [
            [0, 1, *([2] * 10)],
            [0, 1, *([2] * 10)],
            [3, 4, *([5] * 10)],
        ]
    )
    padding = torch.zeros_like(labels, dtype=torch.bool)
    padding[0, :2] = True
    padding[1, 0] = True
    target, history_ids, history_valid = policy._implicit_transition_targets(
        {"subtask_index": labels, "subtask_index_is_pad": padding}
    )
    assert target.tolist() == [2, 2, 5]
    assert history_ids.tolist() == [[0, 1], [0, 1], [3, 4]]
    assert history_valid.tolist() == [[False, False], [False, True], [True, True]]

    policy.config = SimpleNamespace(
        n_action_steps=1,
        implicit_fast_ki_enabled=True,
        action_feature=SimpleNamespace(shape=(1,)),
        adapt_to_pi_aloha=False,
    )
    policy.reset()
    policy._ensure_implicit_transition_history(2, torch.device("cpu"))
    assert not policy._implicit_transition_history_valid.any()
    selected_logits = torch.zeros(2, len(ATOMIC_SKILLS))
    selected_logits[0, 2] = 1
    selected_logits[1, 4] = 1
    policy.model = SimpleNamespace(
        sample_actions=lambda *args, **kwargs: (torch.zeros(2, 1, 1), selected_logits)
    )
    policy.prepare_images = lambda batch, current_phase=None: ([], [])
    policy.prepare_state = lambda batch: batch[OBS_STATE]
    policy._get_action_chunk(
        {
            OBS_STATE: torch.zeros(2, 1),
            "observation.language.tokens": torch.zeros(2, 1, dtype=torch.long),
            "observation.language.attention_mask": torch.ones(2, 1, dtype=torch.bool),
        },
        transition_history_ids=policy._implicit_transition_history_ids,
        transition_history_valid=policy._implicit_transition_history_valid,
    )
    assert policy._implicit_transition_history_ids.tolist() == [[0, 2], [0, 4]]
    assert policy._implicit_transition_history_valid.tolist() == [[False, True], [False, True]]
    policy.reset()
    policy._ensure_implicit_transition_history(2, torch.device("cpu"))
    policy._append_implicit_transition_history(torch.tensor([1, 2]))
    policy._append_implicit_transition_history(torch.tensor([3, 4]))
    policy._append_implicit_transition_history(torch.tensor([5, 0]))
    assert policy._implicit_transition_history_ids.tolist() == [[3, 5], [4, 0]]
    assert policy._implicit_transition_history_valid.tolist() == [[True, True], [True, True]]
    policy.reset()
    assert policy._implicit_transition_history_ids is None
    assert policy._implicit_transition_history_valid is None


def test_implicit_transition_focal_loss_and_validation():
    implicit_config = {
        "chunk_size": 10,
        "n_action_steps": 5,
        "atomic_data_enabled": True,
        "atomic_sgmoe_enabled": True,
        "implicit_fast_ki_enabled": True,
        "atomic_subtask_to_skill": list(range(6)),
        "pretrained_path": Path("local-checkpoint"),
    }
    config = SmolVLAConfig(**implicit_config)
    assert config.implicit_transition_focal_gamma == 2.0
    assert (
        SmolVLAConfig(**implicit_config, implicit_transition_focal_gamma=0).implicit_transition_focal_gamma
        == 0
    )
    for invalid_gamma in (-1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="implicit_transition_focal_gamma"):
            SmolVLAConfig(**implicit_config, implicit_transition_focal_gamma=invalid_gamma)

    cross_entropy = torch.tensor([0.1, 2.0])
    torch.testing.assert_close(
        _implicit_transition_focal_loss(cross_entropy, gamma=0), cross_entropy, rtol=0, atol=0
    )
    relative_factor = _implicit_transition_focal_loss(cross_entropy, gamma=2) / cross_entropy
    assert relative_factor[1] > relative_factor[0]
    assert "_implicit_transition_focal_loss" in inspect.getsource(SmolVLAPolicy.forward)


def test_atomic_classifier_uses_previous_and_current_frame_labels():
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    policy.config = SimpleNamespace(
        chunk_size=4,
        atomic_classifier_enabled=True,
        atomic_subtask_to_skill=list(range(6)),
    )
    target, previous = policy._atomic_classifier_targets(
        {
            "subtask_index": torch.tensor([[0, 1, 1, 2, 2], [0, 4, 4, 4, 4]]),
            "subtask_index_is_pad": torch.tensor(
                [[False, False, False, False, False], [True, False, False, False, False]]
            ),
        }
    )
    assert target.tolist() == [1, 4]
    assert previous.tolist() == [0, 6]


def test_atomic_classifier_event_counts_exclude_episode_starts():
    counts = atomic_classifier_event_counts(
        prediction=torch.tensor([1, 2, 3, 3, 0]),
        target=torch.tensor([1, 2, 3, 3, 5]),
        previous_skill=torch.tensor([1, 1, 4, 2, 6]),
    )
    assert counts == {
        "atomic_stay_correct": 1,
        "atomic_stay_total": 2,
        "atomic_switch_tp": 2,
        "atomic_switch_actual": 2,
        "atomic_switch_predicted": 3,
    }


def test_held_out_atomic_metrics_include_implicit_fast_ki():
    source = (Path(__file__).parents[3] / "src/lerobot/scripts/lerobot_train.py").read_text()
    assert 'getattr(active_cfg, "implicit_fast_ki_enabled", False)' in source
    assert source.count("if atomic_eval_metrics_enabled:") == 2


def test_atomic_classifier_prefix_prefill_skips_missing_expert_tokens():
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    prefix = torch.zeros(1, 4, 3)
    mask = torch.ones(1, 4, dtype=torch.bool)
    model.embed_prefix = lambda *args, **kwargs: (prefix, mask, torch.zeros_like(mask), ((0, 1),), (1, 3))

    class PrefixOnlyVLM(nn.Module):
        def forward(self, **kwargs):
            assert kwargs["inputs_embeds"][1] is None
            assert kwargs["use_cache"] is True
            return [prefix], None

    model.vlm_with_expert = PrefixOnlyVLM()
    model.atomic_previous_skill_embedding = nn.Embedding(7, 3)
    model.atomic_classifier = lambda *args: torch.zeros(1, 6)
    logits = model.classify_atomic_skill(None, None, None, None, torch.zeros(1, 2), torch.tensor([6]))
    assert logits.shape == (1, 6)


def test_atomic_sampler_shuffles_each_selected_frame_once_and_resumes_exactly():
    labels = [0, 0, 0, 1, 2, 3, 4, 5, 5]
    absolute_to_relative = {
        **dict(zip(range(100, 104), range(4), strict=True)),
        **dict(zip(range(110, 115), range(4, 9), strict=True)),
    }

    def make_sampler():
        return AtomicSkillSampler(
            [100, 104, 110],
            [104, 110, 115],
            subtask_indices=labels,
            subtask_to_skill=list(range(6)),
            episode_indices_to_use=[0, 2],
            seed=42,
            absolute_to_relative_idx=absolute_to_relative,
            transition_horizon=5,
        )

    sampler = make_sampler()
    first_epoch = list(sampler)
    assert sorted(first_epoch) == list(range(9))
    sampled_skills = [labels[index] for index in first_epoch]
    assert [sampled_skills.count(skill) for skill in range(6)] == [3, 1, 1, 1, 1, 2]
    assert first_epoch == list(make_sampler())

    resumed = make_sampler()
    resumed.load_state_dict({"epoch": 0, "start_index": 4})
    assert list(resumed) == first_epoch[4:]

    with pytest.raises(ValueError, match="exactly cover"):
        AtomicSkillSampler([0], [6], list(range(6)), [0, 1, 2, 3, 4, 5, 0])
    with pytest.raises(ValueError, match="transition_horizon"):
        AtomicSkillSampler([0], [6], list(range(6)), list(range(6)), transition_horizon=0)


def test_atomic_sampler_aligns_ten_step_targets_and_shuffles_deterministically():
    labels = [*([0] * 11), 1, *([2] * 14), *([3] * 12), 4, 5, *([6] * 12)]
    mapping = [0, 0, 1, 2, 3, 4, 5]
    episode_starts = (0, 26)
    episode_ends = (26, 52)
    transition_horizon = 10

    def make_sampler():
        return AtomicSkillSampler(
            episode_starts,
            episode_ends,
            subtask_indices=labels,
            subtask_to_skill=mapping,
            seed=42,
            anchor_stride=5,
            transition_horizon=transition_horizon,
        )

    sampler = make_sampler()
    starts = set(range(10)) | set(range(26, 36))
    switches = {
        index
        for episode_start, episode_end in zip(episode_starts, episode_ends, strict=True)
        for index in range(episode_start + transition_horizon, episode_end)
        if mapping[labels[index]] != mapping[labels[index - transition_horizon]]
    }
    stays = {10, 22, 36, 50}
    assert switches == set(range(12, 22)) | set(range(38, 50))
    expected = [index for index in range(52) if index not in {11, 23, 24, 25, 37, 51}]
    assert sampler.indices == expected
    assert sampler.retained_candidate_counts == {"start": 20, "switch": 22, "stay": 4}
    assert switches <= set(sampler.indices)
    assert stays <= set(sampler.indices)
    assert labels[10] != labels[11] and mapping[labels[10]] == mapping[labels[11]]
    assert 11 not in switches and 11 not in sampler.indices
    assert starts <= set(sampler.indices)
    assert mapping[labels[25]] != mapping[labels[26]]
    assert set(range(26, 36)).isdisjoint(switches)
    assert {11, 23, 24, 25, 37, 51}.isdisjoint(sampler.indices)

    first_epoch = list(sampler)
    assert sorted(first_epoch) == expected
    assert first_epoch != expected
    assert first_epoch == list(make_sampler())
    resumed = make_sampler()
    resumed.load_state_dict({"epoch": 0, "start_index": 4})
    assert list(resumed) == first_epoch[4:]


def test_atomic_classifier_sampler_draws_current_boundaries_75_25():
    labels = [0, 0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
    kwargs = {
        "dataset_from_indices": [0],
        "dataset_to_indices": [len(labels)],
        "subtask_indices": labels,
        "subtask_to_skill": [0, 0, 1, 2, 3, 4, 5],
        "seed": 42,
        "classifier_event_sampling": True,
    }
    baseline = AtomicSkillSampler(**kwargs)
    sampler = AtomicSkillSampler(**kwargs, anchor_stride=99, transition_horizon=5)
    samples = list(sampler)

    assert samples == list(baseline)
    assert sampler.classifier_candidate_counts == {"stay": 7, "start": 1, "switch": 5}
    assert set(sampler.event_candidates) == {0, 3, 5, 7, 9, 11}
    assert all(sample in sampler.event_candidates for sample in samples[3::4])
    assert all(sample not in sampler.event_candidates for offset in range(3) for sample in samples[offset::4])


def test_select_action_forwards_atomic_skill(monkeypatch):
    from lerobot.policies.smolvla import modeling_smolvla

    monkeypatch.setattr(modeling_smolvla, "populate_queues", lambda queues, batch, exclude_keys=None: queues)
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    policy.config = SimpleNamespace(n_action_steps=1, rtc_config=None)
    policy._queues = {ACTION: deque(maxlen=1)}
    policy.eval = lambda: None
    policy._prepare_batch = lambda batch: batch
    seen = []

    def get_chunk(batch, noise=None, **kwargs):
        seen.append(kwargs["atomic_skill_id"])
        return torch.zeros(1, 1, 1)

    policy._get_action_chunk = get_chunk
    skill = torch.tensor([4])
    policy.select_action({OBS_STATE: torch.zeros(1, 1)}, atomic_skill_id=skill)
    assert seen == [skill]


def test_select_action_uses_frozen_planner_at_each_queue_refill(monkeypatch):
    from lerobot.policies.smolvla import modeling_smolvla

    monkeypatch.setattr(modeling_smolvla, "populate_queues", lambda queues, batch, exclude_keys=None: queues)
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    policy.config = SimpleNamespace(
        atomic_planner_enabled=True,
        n_action_steps=1,
        rtc_config=None,
    )
    policy._queues = {ACTION: deque(maxlen=1)}
    policy.eval = lambda: None
    policy._prepare_batch = lambda batch: batch
    planned = []
    routed = []

    def replan(batch):
        skill = torch.tensor([len(planned)])
        planned.append(skill)
        return skill

    def get_chunk(batch, noise=None, **kwargs):
        routed.append(kwargs["atomic_skill_id"])
        return torch.zeros(1, 1, 1)

    policy.replan_atomic_skill = replan
    policy._get_action_chunk = get_chunk
    batch = {OBS_STATE: torch.zeros(1, 1)}
    policy.select_action(batch)
    policy.select_action(batch)
    assert routed == planned and len(planned) == 2


@pytest.mark.parametrize(
    "raw",
    [
        "invalid",
        "pick place",
        '{"skill":"pick"}',
        "pick\nplace",
    ],
)
def test_atomic_planner_parser_rejects_non_skill_word(raw):
    with pytest.raises(ValueError):
        parse_atomic_planner_output(raw)


def test_atomic_planner_uses_skill_history_and_records_predictions(caplog):
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    policy.config = SimpleNamespace(
        image_features={"observation.images.main": None, "observation.images.wrist": None},
        n_action_steps=5,
    )
    policy._queues = {ACTION: deque([torch.ones(1, 1)], maxlen=5)}
    policy._atomic_planner_skill = None
    policy._atomic_planner_consecutive_failures = 0
    policy.atomic_planner_history = []
    outputs = iter(
        [
            "pick",
            "pick",
            "place",
            "invalid",
            "invalid",
        ]
    )
    prompts = []

    def generate(images, prompt):
        prompts.append(prompt)
        return next(outputs)

    generator = SimpleNamespace(generate_atomic_planner_output=generate)
    policy.model = SimpleNamespace(vlm_with_expert=generator)
    batch = {
        OBS_STATE: torch.zeros(1, 8),
        "task": ["put the mug on the plate"],
        "observation.images.main": torch.zeros(1, 3, 8, 8),
        "observation.images.wrist": torch.zeros(1, 3, 8, 8),
    }

    with caplog.at_level("INFO"):
        assert policy.replan_atomic_skill(batch).item() == 0
        assert not policy._queues[ACTION]
        policy._queues[ACTION].append(torch.ones(1, 1))
        assert policy.replan_atomic_skill(batch).item() == 0
        assert len(policy._queues[ACTION]) == 1
        assert policy.replan_atomic_skill(batch).item() == 1
        assert not policy._queues[ACTION]
        assert policy.replan_atomic_skill(batch).item() == 1
        with pytest.raises(AtomicPlannerEpisodeFailure):
            policy.replan_atomic_skill(batch)
    assert "Executed skill history: none" in prompts[0]
    assert "Executed skill history: pick" in prompts[1]
    assert "Executed skill history: pick, pick" in prompts[2]
    assert "pick: the gripper is not holding the target and must grasp it" in prompts[0]
    assert "place: the gripper is holding the target" in prompts[0]
    assert [record.message for record in caplog.records] == [
        "Atomic planner predicted skill: pick",
        "Atomic planner predicted skill: pick",
        "Atomic planner predicted skill: place",
    ]
    assert [entry["parse_failure"] for entry in policy.atomic_planner_history] == [
        False,
        False,
        False,
        True,
        True,
    ]


def test_first_atomic_planner_parse_failure_ends_episode():
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    policy.config = SimpleNamespace(image_features={"image": None}, n_action_steps=5)
    policy._queues = {ACTION: deque(maxlen=5)}
    policy._atomic_planner_skill = None
    policy._atomic_planner_consecutive_failures = 0
    policy.atomic_planner_history = []
    policy.model = SimpleNamespace(
        vlm_with_expert=SimpleNamespace(generate_atomic_planner_output=lambda images, prompt: "invalid")
    )
    with pytest.raises(AtomicPlannerEpisodeFailure):
        policy.replan_atomic_skill(
            {OBS_STATE: torch.zeros(1, 1), "task": ["task"], "image": torch.zeros(1, 3, 4, 4)}
        )


def test_atomic_planner_loads_full_model_once_without_registering(monkeypatch):
    model = SmolVLMWithExpertModel.__new__(SmolVLMWithExpertModel)
    nn.Module.__init__(model)
    model.vlm = nn.Linear(1, 1, bias=False)
    model.vlm.requires_grad_(False)
    model.model_id = "planner"
    model._atomic_planner = None
    planner = nn.Linear(1, 1)
    processor = object()
    loads = []

    monkeypatch.setattr(
        smolvlm_with_expert.AutoModelForImageTextToText,
        "from_pretrained",
        lambda *args, **kwargs: loads.append((args, kwargs)) or planner,
    )
    monkeypatch.setattr(
        smolvlm_with_expert.AutoProcessor,
        "from_pretrained",
        lambda *_args, **_kwargs: processor,
    )

    assert model._get_atomic_planner() == (planner, processor)
    assert model._get_atomic_planner() == (planner, processor)
    assert len(loads) == 1
    assert not any(parameter.requires_grad for parameter in planner.parameters())
    assert planner not in model.children()


def test_train_expert_only_can_train_only_the_vision_encoder():
    model = SmolVLMWithExpertModel.__new__(SmolVLMWithExpertModel)
    nn.Module.__init__(model)
    model.vlm = nn.Module()
    model.vlm.model = nn.Module()
    model.vlm.model.vision_model = nn.Linear(2, 2)
    model.vlm.model.text_model = nn.Linear(2, 2)
    model.lm_expert = nn.Linear(2, 2)
    model.train_expert_only = True
    model.freeze_vision_encoder = False

    model.set_requires_grad()

    assert all(parameter.requires_grad for parameter in model.vlm.model.vision_model.parameters())
    assert not any(parameter.requires_grad for parameter in model.vlm.model.text_model.parameters())
    model.train()
    assert model.vlm.training is False
    assert model.vlm.model.text_model.training is False
    assert model.vlm.model.vision_model.training is True
    model.eval()
    assert model.vlm.model.vision_model.training is False

    model.freeze_vision_encoder = True
    model.set_requires_grad()
    model.train()
    assert not any(parameter.requires_grad for parameter in model.vlm.model.vision_model.parameters())
    assert model.vlm.model.vision_model.training is False
