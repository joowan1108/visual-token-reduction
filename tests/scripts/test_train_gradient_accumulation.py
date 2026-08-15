import pytest
import torch
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin

from lerobot.configs.train import GradientAccumulationConfig
from lerobot.scripts.lerobot_train import update_policy
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker


class TinyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, batch):
        loss = self.weight * batch.mean()
        return loss, {}


class StepCounter:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1


def test_gradient_accumulation_steps_once_per_effective_batch():
    accelerator = Accelerator(
        cpu=True,
        gradient_accumulation_plugin=GradientAccumulationPlugin(num_steps=2, sync_with_dataloader=False),
    )
    policy = TinyPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    scheduler = StepCounter()
    policy, optimizer = accelerator.prepare(policy, optimizer)
    tracker = MetricsTracker(
        batch_size=1,
        num_frames=2,
        num_episodes=1,
        metrics={
            "loss": AverageMeter("loss", ":.3f"),
            "grad_norm": AverageMeter("grad_norm", ":.3f"),
            "lr": AverageMeter("lr", ":.3e"),
            "update_s": AverageMeter("update_s", ":.3f"),
        },
        accelerator=accelerator,
    )

    update_policy(tracker, policy, torch.ones(1), optimizer, 1.0, accelerator, scheduler)
    torch.testing.assert_close(accelerator.unwrap_model(policy).weight, torch.tensor(1.0))
    assert scheduler.steps == 0

    update_policy(tracker, policy, torch.ones(1), optimizer, 1.0, accelerator, scheduler)
    torch.testing.assert_close(accelerator.unwrap_model(policy).weight, torch.tensor(0.9))
    assert scheduler.steps == 1


def test_gradient_accumulation_steps_must_be_positive():
    with pytest.raises(ValueError, match="must be >= 1"):
        GradientAccumulationConfig(steps=0)
