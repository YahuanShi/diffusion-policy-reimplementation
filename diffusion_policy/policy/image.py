# Stage 5 — how-to-code.md
# Diffusion policy over RGB image + low-dim observations (the main policy).
# Combines MultiImageObsEncoder (Stage 4) with diffusion (Stage 1).
# Reference: diffusion_policy-main/diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py

import torch
import torch.nn as nn

from diffusion_policy.model.diffusion.scheduler import DDPMScheduler
from diffusion_policy.model.diffusion.unet1d import ConditionalUnet1D
from diffusion_policy.model.vision.encoder import MultiImageObsEncoder
from diffusion_policy.dataset.normalizer import LinearNormalizer


class DiffusionUnetImagePolicy(nn.Module):
    def __init__(self, obs_encoder: MultiImageObsEncoder,
                 model: ConditionalUnet1D, noise_scheduler: DDPMScheduler,
                 horizon, action_dim, n_obs_steps, n_action_steps,
                 num_inference_steps=None):
        super().__init__()
        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.horizon = horizon
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.num_inference_steps = num_inference_steps or noise_scheduler.num_train_timesteps
        self.normalizer = LinearNormalizer()

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # batch: {'image': [B, To, 3, H, W], 'agent_pos': [B, To, 2], 'action': [B, H, 2]}
        # Returns: scalar loss
        ...

    @torch.no_grad()
    def predict_action(self, obs_dict):
        # obs_dict: {'image': [B, To, 3, H, W], 'agent_pos': [B, To, 2]}
        # Returns: actions [B, n_action_steps, action_dim]
        ...
