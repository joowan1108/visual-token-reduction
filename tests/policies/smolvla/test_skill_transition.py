import sys
from collections import deque
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.sampler import SkillLinkingSampler
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla import modeling_smolvla as smolvla_mod
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching
from lerobot.utils.constants import ACTION, OBS_PREFIX, OBS_STATE, REWARD


def test_smolvla_config_skill_linking_defaults_and_subtask_delta_indices():
    config = SmolVLAConfig()
    assert config.skill_linking_enabled is False
    assert config.skill_linking_sampler_enabled is False
    assert config.subtask_delta_indices is None

    enabled = SmolVLAConfig(skill_linking_enabled=True, n_action_steps=2, compile_model=False, rtc_config=None)
    assert enabled.subtask_delta_indices == [0, 1, 2]


def test_smolvla_config_skill_linking_validation_rejects_invalid_settings():
    with pytest.raises(ValueError, match="compile_model=False"):
        SmolVLAConfig(skill_linking_enabled=True, compile_model=True)
    with pytest.raises(ValueError, match="does not support RTC"):
        SmolVLAConfig(
            skill_linking_enabled=True,
            rtc_config=SimpleNamespace(enabled=True),
            compile_model=False,
        )
    with pytest.raises(ValueError, match="num_skills \\+ 2"):
        SmolVLAConfig(skill_linking_enabled=True, skill_transition_class_weights=[0.0, 1.0], compile_model=False)


def test_resolve_delta_timestamps_adds_subtask_index_only_when_enabled():
    ds_meta = SimpleNamespace(
        fps=10,
        features={
            REWARD: {},
            "action": {},
            f"{OBS_PREFIX}.state": {},
            "subtask_index": {},
        },
    )

    disabled = SimpleNamespace(
        reward_delta_indices=None,
        action_delta_indices=[0, 1],
        observation_delta_indices=[0],
        subtask_delta_indices=None,
    )
    assert "subtask_index" not in (resolve_delta_timestamps(disabled, ds_meta) or {})

    enabled = SimpleNamespace(
        reward_delta_indices=None,
        action_delta_indices=[0, 1],
        observation_delta_indices=[0],
        subtask_delta_indices=[0, 1, 2],
    )
    delta_timestamps = resolve_delta_timestamps(enabled, ds_meta)
    assert delta_timestamps["subtask_index"] == [0.0, 0.1, 0.2]


def test_skill_linking_sampler_transition_class_counts_and_weights():
    sampler = SkillLinkingSampler(
        dataset_from_indices=[0, 4],
        dataset_to_indices=[4, 8],
        subtask_indices=[1, 1, 2, 2, 2, 2, 1, 1],
        action_horizon=1,
        shuffle=False,
    )

    counts = sampler.transition_class_counts(num_skills=3)
    weights = sampler.transition_class_weights(num_skills=3)

    assert counts == [0, 3, 3, 2, 2]
    assert weights[0] == 0.0
    expected = torch.tensor([1 / (3**0.5), 1 / (3**0.5), 1 / (2**0.5), 1 / (2**0.5)], dtype=torch.float32)
    expected = torch.clamp(expected / expected.mean(), 0.25, 4.0)
    torch.testing.assert_close(torch.tensor(weights[1:], dtype=torch.float32), expected)


def test_skill_linking_sampler_one_boundary_75_25_order_and_resume():
    kwargs = {
        "dataset_from_indices": [0],
        "dataset_to_indices": [7],
        "subtask_indices": [1, 1, 1, 2, 2, 2, 2],
        "action_horizon": 2,
        "shuffle": True,
        "seed": 7,
    }
    sampler = SkillLinkingSampler(**kwargs)

    assert sampler.atomic_candidates == [0, 3, 4]
    assert sampler.boundary_candidates == [1]
    assert sampler.start_candidates == [0]
    assert sampler.done_candidates == [5]
    assert sampler.event_candidates == [0, 1, 5]

    epoch = list(sampler)
    atomic_draws = [sample for position, sample in enumerate(epoch) if position % 4 < 3]
    event_draws = epoch[3::4]
    assert len(epoch) == 12
    assert len(atomic_draws) == 9
    assert len(event_draws) == 3
    assert set(atomic_draws) == set(sampler.atomic_candidates)
    assert set(event_draws) == set(sampler.event_candidates)

    repeated = SkillLinkingSampler(**kwargs)
    assert list(repeated) == epoch

    resumed = SkillLinkingSampler(**kwargs)
    resumed.load_state_dict({"epoch": 0, "start_index": 3})
    assert list(resumed) == epoch[3:]


def test_skill_transition_targets_cover_start_continue_switch_and_done():
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    policy.config = SimpleNamespace(skill_linking_enabled=True, skill_linking_num_skills=16, n_action_steps=2)

    current_skill, target = policy._build_skill_transition_targets(
        {
            "frame_index": torch.tensor([0, 3, 4, 8]),
            "subtask_index": torch.tensor(
                [
                    [1, 1, 2],
                    [5, 5, 5],
                    [7, 7, 9],
                    [4, 4, 0],
                ]
            ),
            "subtask_index_is_pad": torch.tensor(
                [
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, True],
                ]
            ),
        }
    )

    assert current_skill.tolist() == [16, 5, 7, 4]
    assert target.tolist() == [2, 16, 9, 17]


def test_skill_transition_target_validation_rejects_bad_shapes_and_ids():
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    policy.config = SimpleNamespace(skill_linking_enabled=True, skill_linking_num_skills=16, n_action_steps=2)

    with pytest.raises(ValueError, match="rank-2"):
        policy._build_skill_transition_targets(
            {
                "frame_index": torch.tensor([0]),
                "subtask_index": torch.tensor([1, 1, 2]),
                "subtask_index_is_pad": torch.tensor([[False, False, False]]),
            }
        )
    with pytest.raises(ValueError, match="same shape"):
        policy._build_skill_transition_targets(
            {
                "frame_index": torch.tensor([0]),
                "subtask_index": torch.tensor([[1, 1, 2]]),
                "subtask_index_is_pad": torch.tensor([[False, False]]),
            }
        )
    with pytest.raises(ValueError, match=r"\[1, 15\]"):
        policy._build_skill_transition_targets(
            {
                "frame_index": torch.tensor([1]),
                "subtask_index": torch.tensor([[0, 1, 2]]),
                "subtask_index_is_pad": torch.tensor([[False, False, False]]),
            }
        )


def test_skill_embedding_broadcasts_and_preserves_bfloat16_path():
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        skill_linking_enabled=True,
        skill_linking_num_skills=16,
        chunk_size=2,
        min_period=4e-3,
        max_period=4.0,
    )
    model.vlm_with_expert = SimpleNamespace(expert_hidden_size=4)
    model.action_in_proj = nn.Linear(3, 4, bias=False, dtype=torch.bfloat16)
    model.action_time_mlp_in = nn.Linear(8, 4, bias=False, dtype=torch.bfloat16)
    model.action_time_mlp_out = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
    model.skill_embedding = nn.Embedding(17, 4, dtype=torch.bfloat16)
    with torch.no_grad():
        model.action_in_proj.weight.zero_()
        model.action_time_mlp_in.weight.zero_()
        model.action_time_mlp_out.weight.zero_()
        model.skill_embedding.weight.zero_()
        model.skill_embedding.weight[1].fill_(1)
        model.skill_embedding.weight[2].fill_(2)

    noisy_actions = torch.zeros(2, 2, 3, dtype=torch.bfloat16)
    timestep = torch.tensor([0.25, 0.25], dtype=torch.float32)
    embs, _, _ = model.embed_suffix(noisy_actions, timestep, current_skill=torch.tensor([1, 2]))

    assert embs.dtype == torch.bfloat16
    assert embs[0].float().unique().tolist() == [1.0]
    assert embs[1].float().unique().tolist() == [2.0]


def test_select_action_applies_pending_skill_after_exact_horizon(monkeypatch):
    monkeypatch.setattr(smolvla_mod, "populate_queues", lambda queues, batch, exclude_keys=None: queues)

    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    policy.config = SimpleNamespace(
        skill_linking_enabled=True,
        skill_linking_num_skills=16,
        n_action_steps=2,
        rtc_config=None,
    )
    policy._queues = {ACTION: deque(maxlen=2)}
    policy._current_skill = 16
    policy._pending_skill = None
    policy.eval = lambda: None
    policy._prepare_batch = lambda batch: batch
    recorded = []
    pending_logits = [
        torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 5.0] + [0.0] * 12, [0.0] * 17 + [6.0]]),
        torch.tensor([[0.0] * 16 + [4.0, 0.0], [0.0] * 16 + [4.0, 0.0]]),
    ]
    action_chunks = [
        (torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]), pending_logits[0]),
        (torch.tensor([[[5.0], [6.0]], [[7.0], [8.0]]]), pending_logits[1]),
    ]

    def fake_get_action_chunk(batch, noise=None, current_skill=None, **kwargs):
        recorded.append(current_skill.clone())
        actions, logits = action_chunks[len(recorded) - 1]
        policy._pending_skill = logits.argmax(dim=-1)
        return actions

    policy._get_action_chunk = fake_get_action_chunk

    batch = {OBS_STATE: torch.zeros(2, 1)}
    first = policy.select_action(batch)
    second = policy.select_action(batch)
    third = policy.select_action(batch)

    assert first.tolist() == [[1.0], [3.0]]
    assert second.tolist() == [[2.0], [4.0]]
    assert third.tolist() == [[5.0], [7.0]]
    assert recorded[0].tolist() == [16, 16]
    assert recorded[1].tolist() == [5, 16]


def test_pending_reserved_zero_continue_and_done_preserve_current_skill():
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    policy.config = SimpleNamespace(skill_linking_enabled=True, skill_linking_num_skills=16)
    policy._current_skill = torch.tensor([4, 5, 6, 7])
    policy._pending_skill = torch.tensor([0, 16, 17, 3])

    policy._apply_pending_skill()

    assert policy._current_skill.tolist() == [4, 5, 6, 3]
    assert policy._pending_skill is None


def test_reset_clears_skill_state_and_diagnostics():
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    policy.config = SimpleNamespace(n_action_steps=2)
    policy._current_skill = torch.tensor([4])
    policy._pending_skill = torch.tensor([5])
    policy._last_transition_prediction = torch.tensor([5])

    policy.reset()

    assert policy._current_skill is None
    assert policy._pending_skill is None
    assert policy._last_transition_prediction is None
    assert len(policy._queues[ACTION]) == 0


def test_disabled_behavior_stays_passthrough(monkeypatch):
    monkeypatch.setattr(smolvla_mod, "populate_queues", lambda queues, batch, exclude_keys=None: queues)

    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    policy.config = SimpleNamespace(skill_linking_enabled=False)
    policy.eval = lambda: None
    policy._prepare_batch = lambda batch: batch
    policy._queues = {ACTION: deque(maxlen=1)}
    policy._get_action_chunk = lambda batch, noise=None, **kwargs: torch.tensor([[[9.0]]])

    chunk = policy.predict_action_chunk({OBS_STATE: torch.zeros(1, 1)})
    assert chunk.tolist() == [[[9.0]]]


def test_skill_linking_load_whitelist_allows_only_new_modules(monkeypatch):
    delegated = []

    def fake_delegate(cls, model, model_file, map_location, strict):
        delegated.append((model_file, map_location, strict))
        return "delegated"

    monkeypatch.setattr(PreTrainedPolicy, "_load_as_safetensor", classmethod(fake_delegate))

    disabled_model = SimpleNamespace(config=SimpleNamespace(skill_linking_enabled=False))
    assert SmolVLAPolicy._load_as_safetensor(disabled_model, "m.safetensors", "cpu", False) == "delegated"

    enabled_strict_model = SimpleNamespace(config=SimpleNamespace(skill_linking_enabled=True))
    assert SmolVLAPolicy._load_as_safetensor(enabled_strict_model, "m.safetensors", "cpu", True) == "delegated"

    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(skill_linking_enabled=True)
            self._state = {
                "shared.weight": torch.zeros(2, 2),
                "model.skill_embedding.weight": torch.zeros(17, 4),
                "model.transition_head.weight": torch.zeros(18, 4),
                "model.transition_head.bias": torch.zeros(18),
            }

        def state_dict(self):
            return self._state

    monkeypatch.setattr(
        smolvla_mod,
        "_list_safetensor_keys",
        lambda model_file, map_location: {"shared.weight"},
    )
    monkeypatch.setitem(sys.modules, "lerobot.policies.utils", SimpleNamespace(log_model_loading_keys=lambda *args: None))
    monkeypatch.setattr(
        smolvla_mod,
        "_load_safetensor_model",
        lambda model, model_file, map_location, strict: (
            ["model.skill_embedding.weight", "model.transition_head.weight", "model.transition_head.bias"],
            [],
        ),
    )

    model = FakeModel()
    assert SmolVLAPolicy._load_as_safetensor(model, "m.safetensors", "cpu", False) is model


def test_skill_linking_load_whitelist_rejects_partial_skill_checkpoint(monkeypatch):
    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(skill_linking_enabled=True)
            self._state = {
                "shared.weight": torch.zeros(2, 2),
                "model.skill_embedding.weight": torch.zeros(17, 4),
                "model.transition_head.weight": torch.zeros(18, 4),
                "model.transition_head.bias": torch.zeros(18),
            }

        def state_dict(self):
            return self._state

    monkeypatch.setattr(
        smolvla_mod,
        "_list_safetensor_keys",
        lambda model_file, map_location: {
            "shared.weight",
            "model.skill_embedding.weight",
        },
    )
    monkeypatch.setitem(sys.modules, "lerobot.policies.utils", SimpleNamespace(log_model_loading_keys=lambda *args: None))

    with pytest.raises(ValueError, match="Partial skill-linking checkpoint"):
        SmolVLAPolicy._load_as_safetensor(FakeModel(), "m.safetensors", "cpu", False)
