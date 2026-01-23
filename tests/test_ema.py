"""
Unit tests for EMAModel.

Validates:
  - EMA parameters move toward new model parameters over time
  - copy_to produces exact match between EMA and target model
  - Buffers (non-parameter tensors) are correctly copied, not EMA-smoothed
  - decay increases toward 1 as optimization_step grows
"""

import torch
import torch.nn as nn
from diffusion_policy.model.diffusion.ema import EMAModel


def test_ema_tracks_model():
    model = nn.Linear(4, 2)
    ema = EMAModel(model, power=2 / 3)

    # Push model weights toward 1
    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.fill_(1.0)

    for _ in range(200):
        ema.step(model)

    ema_weight = ema.averaged_model.weight
    assert ema_weight.mean().item() > 0.9, "EMA should track model after enough steps"


def test_copy_to_exact():
    model = nn.Linear(4, 2)
    ema = EMAModel(model)
    for _ in range(20):
        with torch.no_grad():
            model.weight.add_(torch.randn_like(model.weight) * 0.1)
        ema.step(model)

    target = nn.Linear(4, 2)
    ema.copy_to(target)

    assert torch.allclose(target.weight, ema.averaged_model.weight)
    assert torch.allclose(target.bias, ema.averaged_model.bias)


def test_decay_increases():
    model = nn.Linear(4, 2)
    ema = EMAModel(model, power=2 / 3)
    decays = []
    for _ in range(50):
        ema.step(model)
        decays.append(ema.decay)

    assert decays[-1] > decays[0], "EMA decay should increase toward 1 over time"
    assert decays[-1] <= 1.0


def test_step_count():
    model = nn.Linear(4, 2)
    ema = EMAModel(model)
    for i in range(15):
        ema.step(model)
    assert ema.optimization_step == 15


def test_buffers_copied():
    class ModelWithBuffer(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2)
            self.register_buffer("running_mean", torch.zeros(2))

        def forward(self, x):
            return self.linear(x)

    model = ModelWithBuffer()
    ema = EMAModel(model)

    model.running_mean.fill_(5.0)
    ema.step(model)

    assert torch.allclose(
        ema.averaged_model.running_mean, torch.tensor([5.0, 5.0])
    ), "Buffers should be hard-copied, not EMA-smoothed"
