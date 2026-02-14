"""
Diffusion Policy ↔ robot_sim adapter (evaluation/sim_policy.py).

Skipped when robot_sim (sibling repo ../robot-sim) is not installed, e.g. in CI.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("robot_sim")

from diffusion_policy.dataset.normalizer import LinearNormalizer  # noqa: E402
from diffusion_policy.model.diffusion.scheduler import DDIMScheduler  # noqa: E402
from diffusion_policy.policy.loader import build_policy  # noqa: E402
from evaluation.sim_policy import DiffusionSimPolicy  # noqa: E402

SHAPE_META = {
    "obs": {
        "observation_images_agentview": {"shape": (3, 32, 32), "type": "rgb"},
        "observation_images_wrist": {"shape": (3, 32, 32), "type": "rgb"},
        "observation_state": {"shape": (9,), "type": "low_dim"},
    },
    "action": {"shape": (10,)},
}
CONFIG = {
    "horizon": 8,
    "n_obs_steps": 2,
    "n_action_steps": 4,
    "down_dims": [16, 32],
    "diffusion_step_embed_dim": 16,
    "num_train_timesteps": 10,
}


def _policy():
    torch.manual_seed(0)
    p = build_policy(
        SHAPE_META, CONFIG, scheduler=DDIMScheduler(10), num_inference_steps=2
    )
    n = LinearNormalizer()
    n.fit({"action": torch.rand(50, 10), "observation_state": torch.rand(50, 9)})
    p.set_normalizer(n)
    return p.eval()


def _obs(t):
    img = np.full((32, 32, 3), t, dtype=np.uint8)
    return {
        "observation.images.agentview": img,
        "observation.images.wrist": img,
        "observation.state": np.full(9, t, dtype=np.float32),
    }


def test_replans_every_n_action_steps(monkeypatch):
    policy = _policy()
    sim = DiffusionSimPolicy(policy, SHAPE_META, device="cpu")
    seen = []
    orig = policy.predict_action

    def spy(batch):
        seen.append(batch["observation_state"][0, :, 0].tolist())
        return orig(batch)

    monkeypatch.setattr(policy, "predict_action", spy)
    sim.reset()
    actions = [sim.act(_obs(t)) for t in range(9)]
    assert all(a.shape == (10,) for a in actions)
    # 9 steps with Ta=4 → predictions at t = 0, 4, 8
    assert len(seen) == 3
    assert seen[0] == [0.0, 0.0]  # history padded with the first frame
    assert seen[1] == [3.0, 4.0]  # last To observations
    assert seen[2] == [7.0, 8.0]


def test_closed_loop_in_simulator():
    from robot_sim.evaluate import evaluate

    result = evaluate(
        DiffusionSimPolicy(_policy(), SHAPE_META, device="cpu"),
        task="lift",
        n_episodes=1,
        image_size=32,
        action_space="abs",
        max_steps=6,
    )
    assert result["n_episodes"] == 1 and result["episodes"][0]["steps"] == 6
