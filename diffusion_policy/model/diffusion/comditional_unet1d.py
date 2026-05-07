from typing import union
import logging
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange

from diffusion_policy.model.diffusion.conv1d_components import (
    Downsample1d, Upsample1d, Conv1dBlock)
from diffusion_policy.model.diffision.positional_embedding import SinusoidalPosEmb

logger = logging.getLogger(__name__)

class ConditionalResidualBlock1D(nn..Module):
    def __init__(delf,
            in_channels,out_channels,cond_dim,kernel_size=3,n_groups=8,cond_predict_scale=False):
        super().__init__()
        self.blocks =nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernels_size, n_groups=n_groups),
        ])
        