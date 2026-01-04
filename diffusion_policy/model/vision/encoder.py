# Stage 4 — how-to-code.md
# Encode RGB images + low-dim observations into a flat feature vector.
# CRITICAL: use GroupNorm, not BatchNorm (incompatible with EMA).
# Reference: diffusion_policy-main/diffusion_policy/model/vision/multi_image_obs_encoder.py

import torch
import torch.nn as nn


def replace_batchnorm_with_groupnorm(module, num_groups=32):
    # Recursively replace all BatchNorm2d → GroupNorm in a module
    ...


class MultiImageObsEncoder(nn.Module):
    def __init__(self, image_shape=(3, 96, 96), lowdim_dim=2, n_obs_steps=2):
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self.backbone = ...      # ResNet-18 with GroupNorm
        self.output_dim = ...    # n_obs_steps * (512 + lowdim_dim)

    def forward(self, obs_dict):
        # obs_dict['image']:     [B, To, 3, H, W]
        # obs_dict['agent_pos']: [B, To, 2]
        # Returns: [B, output_dim]
        ...
