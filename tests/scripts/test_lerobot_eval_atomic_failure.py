from types import SimpleNamespace

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
