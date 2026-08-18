from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from lerobot.policies.pretrained import RolloutEpisodeFailure
from lerobot.scripts import lerobot_eval


def identity(value):
    return value


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

        def __init__(self):
            self.completed = False

        def reset(self, **kwargs):
            return {"observation.state": torch.zeros(1, 8)}, {}

        def call(self, name):
            if name == "_max_episode_steps":
                return [1]
            if name == "task_description":
                return ["put the mug on the plate"]
            if name == "atomic_oracle_attempt":
                return [
                    {
                        "attempt": None
                        if self.completed
                        else {
                            "attempt_id": "place|mug->plate|on|mug|plate",
                            "skill": "place",
                            "target": "mug->plate",
                            "goal": ["on", "mug", "plate"],
                        },
                        "conditions": {"place|mug->plate|on|mug|plate": self.completed},
                    }
                ]
            raise AttributeError(name)

        def step(self, action):
            self.completed = True
            observation = {"observation.state": torch.zeros(1, 8)}
            return (
                observation,
                np.zeros(1),
                np.ones(1, dtype=bool),
                np.zeros(1, dtype=bool),
                {"is_success": np.ones(1, dtype=bool)},
            )

    policy = Policy()
    result = lerobot_eval.rollout(Env(), policy, identity, identity, identity, identity)

    assert policy.skill_ids == [1]
    assert result["success"].tolist() == [[True]]
    assert result["atomic_skill_events"] == [
        [
            {
                "attempt_id": "place|mug->plate|on|mug|plate",
                "skill": "place",
                "target": "mug->plate",
                "goal": ["on", "mug", "plate"],
                "start_step": 0,
                "end_step": 0,
                "success": True,
                "end_reason": "completed",
            }
        ]
    ]


def test_atomic_attempts_use_conditions_and_contiguous_activations():
    place = {
        "attempt_id": "place|mug->plate|on|mug|plate",
        "skill": "place",
        "target": "mug->plate",
        "goal": ["on", "mug", "plate"],
    }
    pick = {"attempt_id": "pick|mug", "skill": "pick", "target": "mug", "goal": []}
    events = []
    active = lerobot_eval._observe_atomic_attempt(
        None,
        events,
        {"attempt": place, "conditions": {place["attempt_id"]: False}},
        0,
        start_new=True,
    )
    active = lerobot_eval._observe_atomic_attempt(
        active,
        events,
        {"attempt": place, "conditions": {place["attempt_id"]: False}},
        1,
        start_new=True,
    )
    active = lerobot_eval._observe_atomic_attempt(
        active,
        events,
        {"attempt": pick, "conditions": {place["attempt_id"]: False, pick["attempt_id"]: False}},
        1,
        start_new=False,
    )
    active = lerobot_eval._observe_atomic_attempt(
        active,
        events,
        {"attempt": pick, "conditions": {pick["attempt_id"]: False}},
        2,
        start_new=True,
    )
    active = lerobot_eval._observe_atomic_attempt(
        active,
        events,
        {"attempt": place, "conditions": {pick["attempt_id"]: True, place["attempt_id"]: False}},
        2,
        start_new=False,
    )
    active = lerobot_eval._observe_atomic_attempt(
        active,
        events,
        {"attempt": place, "conditions": {place["attempt_id"]: False}},
        3,
        start_new=True,
    )
    active = lerobot_eval._observe_atomic_attempt(
        active,
        events,
        {"attempt": None, "conditions": {place["attempt_id"]: True}},
        3,
        start_new=False,
    )

    assert active is None
    assert [(event["skill"], event["success"]) for event in events] == [
        ("place", False),
        ("pick", True),
        ("place", True),
    ]
    assert lerobot_eval._aggregate_atomic_skill_events(events)["place"] == {
        "attempts": 2,
        "successes": 1,
        "success_rate": 0.5,
    }


def test_eval_policy_all_aggregates_atomic_events_by_task_group_and_overall(monkeypatch):
    def event(skill, success):
        return {"skill": skill, "success": success}

    def fake_run_one(task_group, task_id, env, **kwargs):
        events = [[event("pick", True), event("place", task_id == 0)]]
        return (
            task_group,
            task_id,
            {
                "sum_rewards": [float(task_id)],
                "max_rewards": [float(task_id)],
                "successes": [task_id == 0],
                "atomic_planner_skill_timelines": [[]],
                "atomic_skill_events": events,
                "per_skill": lerobot_eval._aggregate_atomic_skill_events(events[0]),
                "video_paths": [],
                "predicted_video_paths": [],
            },
        )

    class Env:
        def close(self):
            pass

    policy = nn.Module()
    policy.atomic_gt_routing = True
    monkeypatch.setattr(lerobot_eval, "run_one", fake_run_one)
    info = lerobot_eval.eval_policy_all(
        {"long": {0: Env(), 1: Env()}},
        policy,
        identity,
        identity,
        identity,
        identity,
        n_episodes=1,
    )

    assert info["per_task"][0]["metrics"]["per_skill"]["pick"]["attempts"] == 1
    assert info["per_group"]["long"]["per_skill"]["place"] == {
        "attempts": 2,
        "successes": 1,
        "success_rate": 0.5,
    }
    assert info["overall"]["per_skill"]["pick"] == {
        "attempts": 2,
        "successes": 2,
        "success_rate": 1.0,
    }
