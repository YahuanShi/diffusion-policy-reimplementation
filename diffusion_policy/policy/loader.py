"""
Policy construction and checkpoint loading — single source of truth for the architecture.

Why this module exists:
  train.py, eval.py and inference.py all need to build the same DiffusionUnetImagePolicy.
  If each of them hardcodes horizon / down_dims / crop_shape, a run trained with
  non-default settings silently gets a mismatched model at load time (no crash —
  ResNet's adaptive pooling accepts any image size — just a distribution shift).

  So training saves its config inside every checkpoint, and loading rebuilds the
  policy from that config. Old checkpoints without a config fall back to DEFAULT_ARCH.

Checkpoint formats:
  step checkpoint  (checkpoint_step*.pt): raw policy weights + EMA state + optimizer, for resuming
  final checkpoint (policy_final.pt):     EMA weights already copied into policy_state_dict
"""

import torch

from diffusion_policy.dataset.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.scheduler import DDIMScheduler, DDPMScheduler
from diffusion_policy.model.vision.encoder import MultiImageObsEncoder
from diffusion_policy.policy.image import DiffusionUnetImagePolicy

# Defaults used by every run before the config was stored in checkpoints
DEFAULT_ARCH = {
    "horizon": 16,
    "n_obs_steps": 2,
    "n_action_steps": 8,
    "down_dims": [256, 512, 1024],
    "diffusion_step_embed_dim": 256,
    "use_group_norm": True,
    "share_rgb_model": False,
    "resize_shape": None,
    "crop_shape": None,
    "num_train_timesteps": 100,
}


def arch_from_config(config):
    """Merge a (possibly partial / missing) training config over DEFAULT_ARCH."""
    arch = dict(DEFAULT_ARCH)
    for k in DEFAULT_ARCH:
        if config and config.get(k) is not None:
            arch[k] = config[k]
    return arch


def build_policy(shape_meta, config=None, scheduler=None, num_inference_steps=None):
    """
    Build an (untrained) DiffusionUnetImagePolicy from shape_meta + training config.

    scheduler: defaults to DDPM (training). Pass a DDIMScheduler for fast inference —
    both share the same trained weights.
    """
    arch = arch_from_config(config)
    crop_shape = tuple(arch["crop_shape"]) if arch["crop_shape"] else None
    resize_shape = tuple(arch["resize_shape"]) if arch["resize_shape"] else None

    encoder = MultiImageObsEncoder(
        shape_meta,
        use_group_norm=arch["use_group_norm"],
        share_rgb_model=arch["share_rgb_model"],
        resize_shape=resize_shape,
        crop_shape=crop_shape,
        # RandomCenterCrop: random in train(), center in eval() — same module either way
        random_crop=(crop_shape is not None),
    )
    if scheduler is None:
        scheduler = DDPMScheduler(num_train_timesteps=arch["num_train_timesteps"])
    return DiffusionUnetImagePolicy(
        obs_encoder=encoder,
        noise_scheduler=scheduler,
        shape_meta=shape_meta,
        horizon=arch["horizon"],
        n_obs_steps=arch["n_obs_steps"],
        n_action_steps=arch["n_action_steps"],
        num_inference_steps=num_inference_steps,
        diffusion_step_embed_dim=arch["diffusion_step_embed_dim"],
        down_dims=list(arch["down_dims"]),
    )


def load_policy(checkpoint_path, device="cpu", num_inference_steps=16, use_ema=True):
    """
    Load a trained policy for inference (DDIM sampler, eval mode).

    Works for both checkpoint formats: for step checkpoints the EMA weights are
    taken from ema_state_dict (policy_state_dict there holds the raw training weights).

    Returns: (policy, shape_meta, config)
    """
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    shape_meta = payload["shape_meta"]
    config = payload.get("config") or {}
    arch = arch_from_config(config)

    normalizer = LinearNormalizer()
    normalizer.load_state_dict(payload["normalizer_state_dict"])

    # DDIM at inference: 16 deterministic steps instead of 100 DDPM steps (~6x faster)
    scheduler = DDIMScheduler(num_train_timesteps=arch["num_train_timesteps"])
    policy = build_policy(
        shape_meta, config, scheduler=scheduler, num_inference_steps=num_inference_steps
    )
    policy.set_normalizer(normalizer)

    if use_ema and "ema_state_dict" in payload:
        state_dict = payload["ema_state_dict"]["averaged_model"]
    else:
        state_dict = payload["policy_state_dict"]
    policy.load_state_dict(state_dict)

    policy.to(device)
    policy.eval()
    return policy, shape_meta, config
