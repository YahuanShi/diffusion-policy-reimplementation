# Stage 3 — how-to-code.md
# Diffusion policy over low-dimensional state observations (no images).
# Start here to isolate the diffusion logic before adding vision.
# Reference: diffusion_policy-main/diffusion_policy/policy/diffusion_unet_lowdim_policy.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.diffusion.scheduler import DDPMScheduler
from diffusion_policy.model.diffusion.unet1d import ConditionalUnet1D
from diffusion_policy.dataset.normalizer import LinearNormalizer


class DiffusionUnetLowdimPolicy(nn.Module):
    def __init__(self, model: ConditionalUnet1D, noise_scheduler: DDPMScheduler,
                 horizon, obs_dim, action_dim, n_obs_steps, n_action_steps,
                 num_inference_steps=None):
        super().__init__()
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.num_inference_steps = num_inference_steps or noise_scheduler.num_train_timesteps
        self.normalizer = LinearNormalizer()

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # batch: {'obs': [B, To, obs_dim], 'action': [B, H, action_dim]}
        # Returns: scalar loss
        ...

    @torch.no_grad()
    def predict_action(self, obs_dict):
        # obs_dict: {'obs': [B, To, obs_dim]}
        # Returns: actions [B, n_action_steps, action_dim]
        ...
