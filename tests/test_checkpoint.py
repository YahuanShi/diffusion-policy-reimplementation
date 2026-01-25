"""
Checkpoint round-trip tests (diffusion_policy/policy/loader.py + evaluation/offline.py).

Validates:
  - load_policy rebuilds the architecture from the config stored in the checkpoint
  - step checkpoints: inference uses the EMA weights, resume uses the raw weights
  - final checkpoints (EMA already copied in) load identically
  - evaluate_offline returns finite metrics with the right per-dim length
"""

import torch

from diffusion_policy.dataset.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.ema import EMAModel
from diffusion_policy.policy.loader import (
    DEFAULT_ARCH,
    arch_from_config,
    build_policy,
    load_policy,
)
from evaluation.offline import evaluate_offline

SHAPE_META = {
    "obs": {
        "image": {"shape": (3, 32, 32), "type": "rgb"},
        "state": {"shape": (2,), "type": "low_dim"},
    },
    "action": {"shape": (2,)},
}
CONFIG = {
    "horizon": 8,
    "n_obs_steps": 2,
    "n_action_steps": 4,
    "down_dims": [16, 32],
    "diffusion_step_embed_dim": 16,
    "num_train_timesteps": 10,
}


def _make_batch(B=3):
    H, To = CONFIG["horizon"], CONFIG["n_obs_steps"]
    return {
        "image": torch.rand(B, To, 3, 32, 32),
        "state": torch.randn(B, To, 2),
        "action": torch.randn(B, H, 2),
    }


def _trained_policy_and_ema():
    torch.manual_seed(0)
    policy = build_policy(SHAPE_META, CONFIG, num_inference_steps=4)
    normalizer = LinearNormalizer()
    normalizer.fit({"action": torch.randn(100, 2), "state": torch.randn(100, 2)})
    policy.set_normalizer(normalizer)
    ema = EMAModel(policy)
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-2)
    for _ in range(3):
        loss = policy.compute_loss(_make_batch())
        loss.backward()
        opt.step()
        opt.zero_grad()
        ema.step(policy)
    return policy, ema, normalizer, opt


def _same_weights(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return all(torch.equal(sa[k], sb[k]) for k in sa)


def test_arch_defaults_for_old_checkpoints():
    assert arch_from_config(None) == DEFAULT_ARCH
    assert arch_from_config({"horizon": 32})["horizon"] == 32


def test_step_checkpoint_loads_ema_for_inference(tmp_path):
    policy, ema, normalizer, opt = _trained_policy_and_ema()
    path = tmp_path / "checkpoint_step3.pt"
    torch.save(
        {
            "epoch": 0,
            "global_step": 3,
            "policy_state_dict": policy.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "normalizer_state_dict": normalizer.state_dict(),
            "shape_meta": SHAPE_META,
            "config": CONFIG,
        },
        path,
    )
    loaded, shape_meta, config = load_policy(path, num_inference_steps=4)
    assert config == CONFIG
    assert loaded.horizon == 8 and loaded.n_action_steps == 4
    assert _same_weights(loaded, ema.averaged_model)
    assert not _same_weights(loaded, policy)  # raw weights differ from EMA

    # Resume path: raw weights must go back into the trained policy
    resumed = build_policy(SHAPE_META, CONFIG)
    resumed.load_state_dict(torch.load(path, weights_only=False)["policy_state_dict"])
    assert _same_weights(resumed, policy)


def test_final_checkpoint_and_prediction_shape(tmp_path):
    policy, ema, normalizer, _ = _trained_policy_and_ema()
    ema.copy_to(policy)
    path = tmp_path / "policy_final.pt"
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "normalizer_state_dict": normalizer.state_dict(),
            "shape_meta": SHAPE_META,
            "config": CONFIG,
        },
        path,
    )
    loaded, _, _ = load_policy(path, num_inference_steps=4)
    assert _same_weights(loaded, ema.averaged_model)

    batch = _make_batch(B=2)
    obs = {k: v for k, v in batch.items() if k != "action"}
    action = loaded.predict_action(obs)
    assert action.shape == (2, CONFIG["n_action_steps"], 2)


def test_evaluate_offline_metrics():
    policy, _, _, _ = _trained_policy_and_ema()
    policy.train()
    loader = [_make_batch(B=2) for _ in range(3)]
    metrics = evaluate_offline(policy, loader, device="cpu", max_batches=2)
    assert policy.training  # mode restored
    assert metrics["n_samples"] == 4
    assert len(metrics["action_mae_per_dim"]) == 2
    assert all(
        torch.isfinite(torch.tensor(v))
        for v in [metrics["loss"], metrics["action_mse"]]
    )
