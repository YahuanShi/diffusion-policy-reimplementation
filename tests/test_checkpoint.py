"""
Unit tests for checkpoint saving/loading and training-loop helpers.

Validates:
  - load_policy rebuilds the trained image preprocessing (resize + crop) from policy_config
  - load_policy prefers EMA weights over raw weights
  - conflicting --resize/--crop overrides are rejected; legacy checkpoints warn
  - step checkpoints are pruned by step number, not filename order
  - LR schedule warms up linearly, then decays to min_ratio
"""

import os

import pytest
import torch
import torchvision

from diffusion_policy.dataset.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.ema import EMAModel
from diffusion_policy.model.diffusion.scheduler import DDPMScheduler
from diffusion_policy.model.vision.encoder import RandomCenterCrop
from diffusion_policy.policy.checkpoint import build_policy, load_policy
from training.workspace import make_lr_lambda, prune_step_checkpoints

SHAPE_META = {
    "obs": {
        "image": {"shape": (3, 64, 64), "type": "rgb"},
        "agent_pos": {"shape": (2,), "type": "low_dim"},
    },
    "action": {"shape": (2,)},
}
POLICY_CONFIG = {
    "horizon": 8,
    "n_obs_steps": 2,
    "n_action_steps": 4,
    "down_dims": [32, 64],
    "diffusion_step_embed_dim": 32,
    "resize_shape": [48, 48],
    "crop_shape": [40, 40],
    "num_train_timesteps": 20,
    "fps": 10,
}


def _save_checkpoint(path, with_config=True):
    policy = build_policy(
        SHAPE_META, POLICY_CONFIG, DDPMScheduler(num_train_timesteps=20)
    )
    normalizer = LinearNormalizer().fit({"action": torch.randn(50, 2)})
    policy.set_normalizer(normalizer)
    ema = EMAModel(policy)
    # Make EMA weights distinguishable from the raw weights
    with torch.no_grad():
        for p in ema.averaged_model.parameters():
            p.fill_(0.5)
    payload = {
        "policy_state_dict": policy.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "normalizer_state_dict": normalizer.state_dict(),
        "shape_meta": SHAPE_META,
    }
    if with_config:
        payload["policy_config"] = POLICY_CONFIG
    torch.save(payload, path)


def test_load_restores_preprocessing_and_ema(tmp_path):
    path = tmp_path / "ckpt.pt"
    _save_checkpoint(path)
    policy, _, cfg = load_policy(path, num_inference_steps=5)

    transform = policy.obs_encoder.key_transform_map["image"]
    resize = [m for m in transform if isinstance(m, torchvision.transforms.Resize)]
    crop = [m for m in transform if isinstance(m, RandomCenterCrop)]
    assert resize and list(resize[0].size) == [48, 48]
    assert crop and crop[0].size == (40, 40)
    assert not policy.training, "policy must be in eval mode (center crop)"
    assert cfg["n_action_steps"] == 4 and cfg["fps"] == 10

    params = list(policy.model.parameters())
    assert all(torch.all(p == 0.5) for p in params), "EMA weights should be loaded"

    obs = {"image": torch.rand(1, 2, 3, 64, 64), "agent_pos": torch.rand(1, 2, 2)}
    assert policy.predict_action(obs).shape == (1, 4, 2)


def test_load_raw_weights_when_ema_disabled(tmp_path):
    path = tmp_path / "ckpt.pt"
    _save_checkpoint(path)
    policy, _, _ = load_policy(path, use_ema=False)
    assert not all(torch.all(p == 0.5) for p in policy.model.parameters())


def test_conflicting_override_rejected(tmp_path):
    path = tmp_path / "ckpt.pt"
    _save_checkpoint(path)
    with pytest.raises(ValueError, match="conflicts"):
        load_policy(path, crop_shape=(30, 30))
    # A matching override is fine
    load_policy(path, crop_shape=(40, 40))


def test_legacy_checkpoint_uses_explicit_preprocessing(tmp_path):
    # Legacy checkpoints have no policy_config, so the default architecture is used;
    # only resize/crop come from the caller.
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "policy_state_dict": build_policy(
                SHAPE_META, {}, DDPMScheduler(num_train_timesteps=100)
            ).state_dict(),
            "normalizer_state_dict": LinearNormalizer()
            .fit({"action": torch.randn(50, 2)})
            .state_dict(),
            "shape_meta": SHAPE_META,
        },
        path,
    )
    with pytest.warns(UserWarning, match="no policy_config"):
        policy, _, cfg = load_policy(path, crop_shape=(56, 56))
    assert cfg["crop_shape"] == [56, 56]
    crop = [
        m
        for m in policy.obs_encoder.key_transform_map["image"]
        if isinstance(m, RandomCenterCrop)
    ]
    assert crop and crop[0].size == (56, 56)


def test_prune_keeps_newest_by_step_number(tmp_path):
    for step in [5000, 10000, 15000, 20000]:
        (tmp_path / f"checkpoint_step{step}.pt").touch()
    (tmp_path / "checkpoint_epoch100.pt").touch()

    prune_step_checkpoints(str(tmp_path), max_keep=3)

    assert sorted(os.listdir(tmp_path)) == [
        "checkpoint_epoch100.pt",
        "checkpoint_step10000.pt",
        "checkpoint_step15000.pt",
        "checkpoint_step20000.pt",
    ]


def test_lr_warmup_then_cosine():
    f = make_lr_lambda(warmup_steps=10, total_steps=110, min_ratio=0.1)
    assert f(0) == pytest.approx(0.1)
    assert f(9) == pytest.approx(1.0)
    assert f(10) == pytest.approx(1.0)
    assert f(60) == pytest.approx(0.55)
    assert f(110) == pytest.approx(0.1)
    assert f(500) == pytest.approx(0.1), "LR stays at the floor past total_steps"
    warmup = [f(s) for s in range(10)]
    assert warmup == sorted(warmup)
