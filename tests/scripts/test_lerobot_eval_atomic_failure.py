from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from lerobot.policies.pretrained import RolloutEpisodeFailure
from lerobot.scripts import lerobot_eval


def test_rollout_records_policy_requested_episode_failure(monkeypatch):
    monkeypatch.setattr(lerobot_eval, "preprocess_observation", lambda observation: observation)
    monkeypatch.setattr(lerobot_eval, "check_env_attributes_and_types", lambda env: None)

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                atomic_planner_enabled=True,
                action_feature=SimpleNamespace(shape=(7,)),
            )

        def reset(self):
            pass

        def select_action(self, observation):
            raise RolloutEpisodeFailure("invalid first planner response")

    class Env:
        num_envs = 1

        def reset(self, **kwargs):
            return {"observation.state": torch.zeros(1, 8)}, {}

        def call(self, name):
            if name == "_max_episode_steps":
                return [10]
            if name == "task_description":
                return ["put the mug on the plate"]
            raise AttributeError(name)

    identity = lambda value: value
    result = lerobot_eval.rollout(Env(), Policy(), identity, identity, identity, identity)

    assert result["done"].tolist() == [[True]]
    assert result["success"].tolist() == [[False]]
    assert result["policy_failure"].tolist() == [[True]]
    assert result["action"].shape == (1, 1, 7)


def test_rollout_passes_libero_gt_skill_to_policy(monkeypatch):
    monkeypatch.setattr(lerobot_eval, "preprocess_observation", lambda observation: observation)
    monkeypatch.setattr(lerobot_eval, "check_env_attributes_and_types", lambda env: None)

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.atomic_gt_routing = True
            self.skill_ids = []
            self.config = SimpleNamespace(atomic_planner_enabled=False, atomic_classifier_enabled=False)

        def reset(self):
            pass

        def select_action(self, observation, *, atomic_skill_id):
            self.skill_ids.extend(atomic_skill_id.tolist())
            return torch.zeros(1, 7)

    class Env:
        num_envs = 1

        def reset(self, **kwargs):
            return {"observation.state": torch.zeros(1, 8)}, {}

        def call(self, name):
            if name == "_max_episode_steps":
                return [1]
            if name == "task_description":
                return ["put the mug on the plate"]
            if name == "atomic_oracle_skill":
                return ["place"]
            raise AttributeError(name)

        def step(self, action):
            observation = {"observation.state": torch.zeros(1, 8)}
            return (
                observation,
                np.zeros(1),
                np.ones(1, dtype=bool),
                np.zeros(1, dtype=bool),
                {"is_success": np.ones(1, dtype=bool)},
            )

    policy = Policy()
    identity = lambda value: value
    result = lerobot_eval.rollout(Env(), policy, identity, identity, identity, identity)

    assert policy.skill_ids == [1]
    assert result["success"].tolist() == [[True]]
