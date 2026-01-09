"""
Stage 1 validation — run this after implementing scheduler.py and unet1d.py.

Trains a diffusion model on a 2D bimodal distribution (no robot data needed).
Success criterion: the scatter plot toy_samples.png shows two distinct clusters.
If you see one blob in the middle, the denoiser is not learning multimodality.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from diffusion_policy.model.diffusion.scheduler import DDPMScheduler
from diffusion_policy.model.diffusion.unet1d import ConditionalUnet1D


SCALE = 3.0  # data range: normalize [-3, 3] -> [-1, 1]


def sample_bimodal(n):
    """2D points from a mixture of two Gaussians at x=[-2,0] and x=[+2,0]."""
    mode = torch.randint(0, 2, (n,))
    means = torch.tensor([[-2.0, 0.0], [2.0, 0.0]])
    return means[mode] + torch.randn(n, 2) * 0.3


def main():
    H = 4
    scheduler = DDPMScheduler(num_train_timesteps=100)
    model = ConditionalUnet1D(
        input_dim=2, global_cond_dim=None,
        down_dims=[32, 64], diffusion_step_embed_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    num_steps = 5000
    for step in range(num_steps):
        x0_raw = sample_bimodal(256)
        x0 = (x0_raw / SCALE).unsqueeze(1).repeat(1, H, 1)
        noise = torch.randn_like(x0)
        t = torch.randint(0, 100, (256,))
        xt = scheduler.add_noise(x0, noise, t)
        eps_pred = model(xt, t)
        loss = nn.functional.mse_loss(eps_pred, noise)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        if step % (num_steps // 5) == 0:
            print(f"step {step:>5}: loss={loss.item():.4f}")

    # Sampling
    model.eval()
    scheduler.set_timesteps(100)
    x = torch.randn(1000, H, 2)
    with torch.no_grad():
        for t in scheduler.timesteps:
            eps_pred = model(x, t.unsqueeze(0).expand(1000))
            x = scheduler.step(eps_pred, t, x).prev_sample

    pts = (x[:, 0, :] * SCALE).numpy()  # denormalize
    plt.figure(figsize=(5, 5))
    plt.scatter(pts[:, 0], pts[:, 1], alpha=0.3, s=5)
    plt.title("Stage 1 check: should show two clusters at x=-2 and x=+2")
    plt.savefig("experiments/toy_samples.png", dpi=100)
    print("Saved experiments/toy_samples.png")


if __name__ == '__main__':
    main()
