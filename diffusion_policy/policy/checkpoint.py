"""
Policy construction and checkpoint loading — one place that turns a config into a policy.

Why this exists:
  Training, eval.py and inference.py must build *exactly* the same network and image
  preprocessing (resize, crop). If inference rebuilds the encoder by hand and forgets the
  crop, ResNet's adaptive pooling still accepts the image, so nothing crashes — the policy
  just silently sees a different input distribution than it was trained on.

  So training stores `policy_config` inside every checkpoint, and loading rebuilds the
  policy from it. Checkpoints saved before this existed fall back to DEFAULT_POLICY_CONFIG
  plus whatever resize/crop the caller passes explicitly.

Weights:
  Step/epoch checkpoints carry both the raw weights and the EMA weights. EMA weights are
  what should be evaluated (they are what policy_final.pt contains), so load_policy prefers
  them when present.
"""

import warnings

import torch

from diffusion_policy.dataset.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.scheduler import DDIMScheduler
from diffusion_policy.model.vision.encoder import MultiImageObsEncoder
from diffusion_policy.policy.image import DiffusionUnetImagePolicy

# Architecture used by every checkpoint saved before policy_config was stored
DEFAULT_POLICY_CONFIG = {
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
    "fps": None,
}


def _as_tuple(shape):
    return tuple(shape) if shape is not None else None


def build_policy(shape_meta, policy_config, noise_scheduler, num_inference_steps=None):
    """Build encoder + policy from a policy_config dict (see DEFAULT_POLICY_CONFIG)."""
    cfg = {**DEFAULT_POLICY_CONFIG, **policy_config}
    crop_shape = _as_tuple(cfg["crop_shape"])
    encoder = MultiImageObsEncoder(
        shape_meta,
        use_group_norm=cfg["use_group_norm"],
        share_rgb_model=cfg["share_rgb_model"],
        resize_shape=_as_tuple(cfg["resize_shape"]),
        crop_shape=crop_shape,
        # Random crop in train mode, center crop in eval mode — same module either way
        random_crop=(crop_shape is not None),
    )
    return DiffusionUnetImagePolicy(
        obs_encoder=encoder,
        noise_scheduler=noise_scheduler,
        shape_meta=shape_meta,
        horizon=cfg["horizon"],
        n_obs_steps=cfg["n_obs_steps"],
        n_action_steps=cfg["n_action_steps"],
        num_inference_steps=num_inference_steps,
        diffusion_step_embed_dim=cfg["diffusion_step_embed_dim"],
        down_dims=list(cfg["down_dims"]),
    )


def load_policy(
    checkpoint_path,
    device="cpu",
    num_inference_steps=16,
    use_ema=True,
    resize_shape=None,
    crop_shape=None,
):
    """
    Load a checkpoint into an eval-mode policy using a DDIM scheduler.

    resize_shape / crop_shape are only needed for legacy checkpoints that do not store
    policy_config; for new checkpoints they must be omitted or match the stored values.

    Returns: (policy, shape_meta, policy_config)
    """
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    shape_meta = payload["shape_meta"]

    cfg = dict(DEFAULT_POLICY_CONFIG)
    stored = payload.get("policy_config")
    overrides = {"resize_shape": resize_shape, "crop_shape": crop_shape}
    if stored is not None:
        cfg.update(stored)
        for key, value in overrides.items():
            if value is not None and _as_tuple(value) != _as_tuple(cfg[key]):
                raise ValueError(
                    f"--{key.removesuffix('_shape')} {list(value)} conflicts with the "
                    f"checkpoint's training value {cfg[key]}. Omit it — the checkpoint "
                    "already records its preprocessing."
                )
    else:
        warnings.warn(
            f"{checkpoint_path} has no policy_config (saved by an older version). "
            "Using default architecture; pass the training --resize/--crop explicitly.",
            stacklevel=2,
        )
        for key, value in overrides.items():
            if value is not None:
                cfg[key] = list(value)

    normalizer = LinearNormalizer()
    normalizer.load_state_dict(payload["normalizer_state_dict"])

    scheduler = DDIMScheduler(num_train_timesteps=cfg["num_train_timesteps"])
    policy = build_policy(shape_meta, cfg, scheduler, num_inference_steps)
    policy.set_normalizer(normalizer)

    if use_ema and "ema_state_dict" in payload:
        state_dict = payload["ema_state_dict"]["averaged_model"]
    else:
        state_dict = payload["policy_state_dict"]
    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()
    return policy, shape_meta, cfg
