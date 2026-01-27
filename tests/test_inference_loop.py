"""
End-to-end test of the real-robot control loop with a fake robot (no hardware).

Validates:
  - run_inference executes exactly max_steps actions and stops
  - the robot receives a boolean gripper command for every executed action
  - the measured gripper width maps to the training-data convention (1 = open)
  - TrialRecorder writes videos, steps.npz and meta.json for the trial
  - timing statistics are collected for every executed step
"""

import json
import os

import numpy as np
import torch

import inference
from diffusion_policy.dataset.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.scheduler import DDIMScheduler
from diffusion_policy.policy.loader import build_policy
from evaluation.trial_recorder import TrialRecorder

SHAPE_META = {
    "obs": {
        "observation_images_exterior_image_1_left": {
            "shape": (3, 32, 32),
            "type": "rgb",
        },
        "observation_images_wrist_image_left": {"shape": (3, 32, 32), "type": "rgb"},
        "observation_state": {"shape": (7,), "type": "low_dim"},
    },
    "action": {"shape": (7,)},
}
CONFIG = {
    "horizon": 8,
    "n_obs_steps": 2,
    "n_action_steps": 4,
    "down_dims": [16, 32],
    "diffusion_step_embed_dim": 16,
    "num_train_timesteps": 10,
}


class FakeRobot:
    def __init__(self):
        self.commands = []

    def get_obs(self):
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        return {
            "state": np.ones(7, dtype=np.float32),
            "exterior_rgb": img,
            "wrist_rgb": img.copy(),
            "sensor_age_s": 0.001,
        }

    def apply_action(self, action, gripper_open):
        self.commands.append((np.asarray(action).copy(), gripper_open))


def _policy():
    torch.manual_seed(0)
    policy = build_policy(
        SHAPE_META, CONFIG, scheduler=DDIMScheduler(10), num_inference_steps=2
    )
    normalizer = LinearNormalizer()
    normalizer.fit(
        {"action": torch.rand(100, 7), "observation_state": torch.rand(100, 7)}
    )
    policy.set_normalizer(normalizer)
    return policy.eval()


def test_run_inference_with_fake_robot(tmp_path, monkeypatch):
    monkeypatch.setattr(inference.cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(inference.cv2, "pollKey", lambda: -1)
    monkeypatch.setattr(inference.cv2, "destroyAllWindows", lambda: None)

    robot = FakeRobot()
    recorder = TrialRecorder(str(tmp_path), fps=50, meta={"checkpoint": "fake"})
    timing = inference.run_inference(
        policy=_policy(),
        robot=robot,
        shape_meta=SHAPE_META,
        device="cpu",
        steps_per_inference=3,
        max_steps=9,
        frequency=50,
        recorder=recorder,
    )

    assert len(robot.commands) == 9
    assert all(isinstance(g, bool) for _, g in robot.commands)
    assert timing.summary()["obs"]["n"] >= 9

    trial_dir = recorder.close(success=True, notes="fake")
    for name in ("exterior.mp4", "wrist.mp4", "steps.npz", "meta.json"):
        assert os.path.exists(os.path.join(trial_dir, name)), name
    steps = np.load(os.path.join(trial_dir, "steps.npz"))
    assert steps["action"].shape == (9, 7)
    meta = json.load(open(os.path.join(trial_dir, "meta.json")))
    assert meta["success"] is True and meta["n_steps"] == 9


def test_gripper_state_matches_training_convention():
    # Training data: 1 = open (jaws apart), 0 = closed (holding the object)
    assert inference.gripper_state_from_width(inference.GRIPPER_MAX_MM) == 1.0
    assert inference.gripper_state_from_width(0.0) == 0.0
