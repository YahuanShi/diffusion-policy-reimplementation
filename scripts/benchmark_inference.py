"""
Inference latency benchmark — measures DDIM/DDPM denoising speed at different step counts.

Usage:
    python scripts/benchmark_inference.py                    # CPU benchmark
    python scripts/benchmark_inference.py --device cuda      # GPU benchmark
    python scripts/benchmark_inference.py --action_dim 6 --obs_dim 1024

Output example:
    device: cuda  |  action_dim=6  horizon=16  obs_dim=1024
    ┌────────────────┬───────────┬───────────┬──────────┐
    │ scheduler      │ steps     │ latency   │ freq     │
    ├────────────────┼───────────┼───────────┼──────────┤
    │ DDIM           │ 16        │  12.3 ms  │  81 Hz   │
    │ DDIM           │ 32        │  23.1 ms  │  43 Hz   │
    │ DDPM           │ 100       │  78.4 ms  │  13 Hz   │
    └────────────────┴───────────┴───────────┴──────────┘
"""

import argparse
import time

import torch

from diffusion_policy.model.diffusion.scheduler import DDIMScheduler, DDPMScheduler
from diffusion_policy.model.diffusion.unet1d import ConditionalUnet1D


def build_unet(action_dim, obs_dim, horizon, device):
    model = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
    ).to(device)
    model.eval()
    return model


@torch.no_grad()
def benchmark(scheduler_cls, n_steps, model, action_dim, obs_dim, horizon, device, n_warmup=10, n_repeat=50):
    scheduler = scheduler_cls(num_train_timesteps=100)
    scheduler.set_timesteps(n_steps)

    global_cond = torch.randn(1, obs_dim, device=device)

    # Warmup
    for _ in range(n_warmup):
        x = torch.randn(1, horizon, action_dim, device=device)
        for t in scheduler.timesteps:
            eps = model(x, t, global_cond=global_cond)
            x = scheduler.step(eps, t, x).prev_sample

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_repeat):
        x = torch.randn(1, horizon, action_dim, device=device)
        for t in scheduler.timesteps:
            eps = model(x, t, global_cond=global_cond)
            x = scheduler.step(eps, t, x).prev_sample
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    latency_ms = elapsed / n_repeat * 1000
    freq_hz = 1000 / latency_ms
    return latency_ms, freq_hz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--action_dim", type=int, default=6)
    parser.add_argument("--obs_dim", type=int, default=1024)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n_repeat", type=int, default=50)
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    model = build_unet(args.action_dim, args.obs_dim, args.horizon, device)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\ndevice: {device}  |  action_dim={args.action_dim}  horizon={args.horizon}  obs_dim={args.obs_dim}  params={n_params:,}")
    print(f"{'scheduler':<12}  {'steps':>6}  {'latency':>10}  {'freq':>8}")
    print("-" * 44)

    configs = [
        (DDIMScheduler, 16),
        (DDIMScheduler, 32),
        (DDIMScheduler, 64),
        (DDPMScheduler, 100),
    ]

    for cls, steps in configs:
        name = cls.__name__.replace("Scheduler", "")
        lat, freq = benchmark(
            cls, steps, model, args.action_dim, args.obs_dim, args.horizon, device, n_repeat=args.n_repeat
        )
        print(f"{name:<12}  {steps:>6}  {lat:>8.1f} ms  {freq:>6.0f} Hz")

    print()


if __name__ == "__main__":
    main()
