import math
from collections import deque
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.datasets.sampler import AtomicSkillSampler
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import (
    AtomicPlannerEpisodeFailure,
    SmolVLAPolicy,
    parse_atomic_planner_output,
)
from lerobot.policies.smolvla.smolvlm_with_expert import (
    AtomicSkillFFN,
    AtomicSkillRouter,
    SmolVLMWithExpertModel,
)
from lerobot.utils.constants import ACTION, OBS_STATE


def test_atomic_config_keeps_dense_defaults_and_freezes_temporal_contract():
    assert SmolVLAConfig().atomic_sgmoe_enabled is False
    config = SmolVLAConfig(
        chunk_size=10,
        n_action_steps=5,
        atomic_data_enabled=True,
        atomic_sgmoe_enabled=True,
        atomic_subtask_to_skill=list(range(6)),
    )
    assert config.subtask_delta_indices == list(range(10))
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


def test_atomic_boundary_mask_uses_canonical_skill_not_subtask_identity():
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    policy.config = SimpleNamespace(chunk_size=4, atomic_subtask_to_skill=[0, 1, 1, 2, 3, 4, 5])
    skill, action_is_pad = policy._atomic_batch_contract(
        {
            "subtask_index": torch.tensor([[1, 2, 3, 3]]),
            "subtask_index_is_pad": torch.tensor([[False, False, False, True]]),
        }
    )
    assert skill.tolist() == [1]
    assert action_is_pad.tolist() == [[False, False, True, True]]


def test_atomic_sampler_is_balanced_deterministic_and_resumable():
    sampler = AtomicSkillSampler(
        [0],
        [60],
        subtask_indices=[index % 6 for index in range(60)],
        subtask_to_skill=list(range(6)),
        seed=42,
    )
    first_epoch = list(sampler)
    assert [index % 6 for index in first_epoch].count(0) == 10
    assert all([index % 6 for index in first_epoch].count(skill) == 10 for skill in range(6))
    assert first_epoch == list(
        AtomicSkillSampler([0], [60], [index % 6 for index in range(60)], list(range(6)), seed=42)
    )

    resumed = AtomicSkillSampler([0], [60], [index % 6 for index in range(60)], list(range(6)), seed=42)
    resumed.load_state_dict({"epoch": 0, "start_index": 17})
    assert list(resumed) == first_epoch[17:]

    with pytest.raises(ValueError, match="exactly cover"):
        AtomicSkillSampler([0], [6], list(range(6)), [0, 1, 2, 3, 4, 5, 0])


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
    ("raw", "previous"),
    [
        ('{"decision":"continue","skill":"pick"}', None),
        ('{"decision":"continue","skill":"place"}', "pick"),
        ('{"decision":"switch","skill":"pick"}', "pick"),
        ('{"decision":"switch","skill":"pick","extra":1}', None),
        ('{"decision":"switch","decision":"continue","skill":"pick"}', None),
        ('{"decision":"switch","skill":"pick"} trailing', None),
    ],
)
def test_atomic_planner_parser_rejects_non_strict_or_inconsistent_json(raw, previous):
    with pytest.raises(ValueError):
        parse_atomic_planner_output(raw, previous)


def test_atomic_planner_state_machine_switch_continue_and_failures():
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
            '{"decision":"switch","skill":"pick"}',
            '{"decision":"continue","skill":"pick"}',
            '{"decision":"switch","skill":"place"}',
            "invalid",
            "invalid",
        ]
    )
    generator = SimpleNamespace(generate_atomic_planner_output=lambda images, prompt: next(outputs))
    policy.model = SimpleNamespace(vlm_with_expert=generator)
    batch = {
        OBS_STATE: torch.zeros(1, 8),
        "task": ["put the mug on the plate"],
        "observation.images.main": torch.zeros(1, 3, 8, 8),
        "observation.images.wrist": torch.zeros(1, 3, 8, 8),
    }

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
