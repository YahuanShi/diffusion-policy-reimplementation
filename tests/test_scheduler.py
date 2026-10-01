"""
Unit tests for DDPM and DDIM schedulers.

Validates:
  - Forward process endpoints (signal-dominated at t=0, noise-dominated at t=T-1)
  - DDIM is deterministic (same seed → same trajectory)
  - DDPM / DDIM share the same weights and both can run a reverse pass
  - set_timesteps produces correct subsampling
"""

import torch
from diffusion_policy.model.diffusion.scheduler import DDPMScheduler, DDIMScheduler


def test_forward_process_endpoints():
    scheduler = DDPMScheduler(num_train_timesteps=100)
    x0 = torch.zeros(4, 16, 2)
    noise = torch.ones(4, 16, 2)

    xt_early = scheduler.add_noise(x0, noise, torch.full((4,), 0, dtype=torch.long))
    xt_late = scheduler.add_noise(x0, noise, torch.full((4,), 99, dtype=torch.long))

    # At t=0 the signal coefficient is close to 1, so output is close to 0
    assert xt_early.mean().item() < 0.1, "t=0 should be signal-dominated"
    # At t=99 the noise coefficient is close to 1, so output is close to 1
    assert xt_late.mean().item() > 0.9, "t=99 should be noise-dominated"


def test_alphas_cumprod_monotone():
    scheduler = DDPMScheduler(num_train_timesteps=100)
    # alpha_bar should be strictly decreasing
    diffs = scheduler.alphas_cumprod[1:] - scheduler.alphas_cumprod[:-1]
    assert (diffs < 0).all(), "alpha_cumprod must be monotonically decreasing"


def test_ddpm_set_timesteps_subsampling():
    scheduler = DDPMScheduler(num_train_timesteps=100)
    scheduler.set_timesteps(16)
    assert len(scheduler.timesteps) == 16
    assert scheduler.timesteps[0] == 99
    assert scheduler.timesteps[-1] == 0


def test_ddim_deterministic():
    torch.manual_seed(0)
    scheduler = DDIMScheduler(num_train_timesteps=100)
    scheduler.set_timesteps(16)

    torch.manual_seed(42)
    x = torch.randn(1, 16, 2)
    traj_a = x.clone()
    for t in scheduler.timesteps:
        eps = torch.randn_like(traj_a)
        traj_a = scheduler.step(eps, t, traj_a).prev_sample

    torch.manual_seed(42)
    x = torch.randn(1, 16, 2)
    traj_b = x.clone()
    for t in scheduler.timesteps:
        eps = torch.randn_like(traj_b)
        traj_b = scheduler.step(eps, t, traj_b).prev_sample

    assert torch.allclose(traj_a, traj_b), (
        "DDIM must be deterministic given same inputs"
    )


def test_ddpm_oracle_reconstructs_x0():
    scheduler = DDPMScheduler(num_train_timesteps=100)
    x0 = torch.ones(1, 1, 1)
    eps = torch.randn_like(x0)
    t = 50
    xt = scheduler.add_noise(x0, eps, torch.tensor([t]))

    # With the true noise, the internal pred_x0 should equal x0 exactly.
    # pred_x0 = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)
    alpha_t = scheduler.alphas_cumprod[t]
    pred_x0 = (xt - (1 - alpha_t).sqrt() * eps) / alpha_t.sqrt()
    assert torch.allclose(pred_x0, x0, atol=1e-5), (
        "Oracle eps must reconstruct x0 exactly"
    )


def test_ddpm_ddim_compatible_shapes():
    for Scheduler in [DDPMScheduler, DDIMScheduler]:
        s = Scheduler(num_train_timesteps=50)
        s.set_timesteps(8)
        x = torch.randn(2, 16, 4)
        for t in s.timesteps:
            eps = torch.randn_like(x)
            result = s.step(eps, t, x)
            x = result.prev_sample
        assert x.shape == (2, 16, 4)


def test_ddpm_full_schedule_matches_closed_form_posterior():
    # With all T steps, the step must reduce to the textbook DDPM posterior.
    scheduler = DDPMScheduler(num_train_timesteps=100)
    x_t = torch.randn(2, 16, 2)
    eps = torch.randn_like(x_t)
    t = 50
    a_t, a_prev = scheduler.alphas_cumprod[t], scheduler.alphas_cumprod[t - 1]
    pred_x0 = ((x_t - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-1, 1)
    expected_mean = (
        a_prev.sqrt() * scheduler.betas[t] / (1 - a_t) * pred_x0
        + scheduler.alphas[t].sqrt() * (1 - a_prev) / (1 - a_t) * x_t
    )

    torch.manual_seed(0)
    out = scheduler.step(eps, t, x_t).prev_sample
    torch.manual_seed(0)
    expected = expected_mean + scheduler.posterior_variance[
        t
    ].sqrt() * torch.randn_like(x_t)
    assert torch.allclose(out, expected, atol=1e-5)


def test_ddpm_subsampled_marginals():
    # With an oracle noise prediction (x0 = 0), each reverse step must land on the
    # forward-process marginal of the *next scheduled* timestep: std = sqrt(1 - a_t').
    torch.manual_seed(0)
    scheduler = DDPMScheduler(num_train_timesteps=100)
    scheduler.set_timesteps(10)
    ts = scheduler.timesteps
    x = torch.randn(20000, 1, 1) * (1 - scheduler.alphas_cumprod[ts[0]]).sqrt()
    for i, t in enumerate(ts[:-1]):
        eps = x / (1 - scheduler.alphas_cumprod[t]).sqrt()
        x = scheduler.step(eps, t, x).prev_sample
        expected_std = (1 - scheduler.alphas_cumprod[ts[i + 1]]).sqrt().item()
        assert abs(x.std().item() - expected_std) < 0.02 * max(expected_std, 0.1), (
            f"step {i}: std {x.std().item():.4f} != {expected_std:.4f}"
        )
