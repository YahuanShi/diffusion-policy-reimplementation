# Stage 6 — how-to-code.md
# Full training orchestrator: dataset → policy → EMA → optimizer → eval loop.
# Reference: diffusion_policy-main/diffusion_policy/workspace/train_diffusion_unet_hybrid_workspace.py

import copy
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from diffusion_policy.model.diffusion.scheduler import DDPMScheduler
from diffusion_policy.model.diffusion.unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.ema import EMAModel
from diffusion_policy.model.vision.encoder import MultiImageObsEncoder
from diffusion_policy.dataset.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.sampler import SequenceSampler
from diffusion_policy.dataset.normalizer import LinearNormalizer
from diffusion_policy.policy.image import DiffusionUnetImagePolicy


def train(zarr_path, device='cuda', batch_size=64, num_epochs=3050,
          horizon=16, n_obs_steps=2, n_action_steps=8):
    # 1. Build dataset + normalizer
    # 2. Build policy (obs_encoder + unet1d + scheduler)
    # 3. Build EMA shadow copy
    # 4. Build optimizer + cosine LR schedule with warmup
    # 5. Training loop:
    #      - compute_loss → backward → optimizer step → EMA step
    #      - every 50 epochs: copy EMA → eval → save checkpoint
    ...
