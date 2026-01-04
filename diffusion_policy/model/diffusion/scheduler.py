# Stage 1A — how-to-code.md
# Implement the DDPM forward and reverse process from scratch.
# Reference: diffusion_policy-main/diffusion_policy/model/diffusion/ (schedulers)

import torch


class DDPMScheduler:
    def __init__(self, num_train_timesteps=100, beta_start=0.0001,
                 beta_end=0.02, beta_schedule='squaredcos_cap_v2'):
        self.num_train_timesteps = num_train_timesteps
        self.betas = ...
        self.alphas = ...
        self.alphas_cumprod = ...
        self.sqrt_alphas_cumprod = ...
        self.sqrt_one_minus_alphas_cumprod = ...
        self.timesteps = torch.arange(num_train_timesteps - 1, -1, -1)

    def add_noise(self, x0, noise, timesteps):
        # x0: [B, H, D]  noise: [B, H, D]  timesteps: [B]
        # Returns x_t: [B, H, D]
        ...

    def step(self, eps_pred, t, x_t):
        # One reverse step: x_t → x_{t-1}
        # Returns object with .prev_sample attribute
        ...

    def set_timesteps(self, num_inference_steps):
        self.timesteps = torch.arange(num_inference_steps - 1, -1, -1)
