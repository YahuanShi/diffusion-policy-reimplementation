"""
Adapter: DiffusionUnetImagePolicy → robot_sim's Policy interface (reset / act).

robot_sim calls act(obs) once per control step with a single observation. Diffusion
Policy instead conditions on the last To observations and predicts Ta actions at once,
so this adapter keeps
  - an observation deque of length To (filled with copies of the first frame on reset,
    exactly like the real-robot loop in inference.py)
  - an action queue: re-plan only when the queue is empty, then execute the chunk

In simulation there is no wall clock — inference latency does not cost control steps,
so all n_action_steps predicted actions are executed (no latency skipping).

reset(seed) seeds torch with the episode's environment seed, so a rerun of the same
checkpoint reproduces every episode exactly, and a single failing seed can be replayed.

Observation keys: robot_sim emits LeRobot names ("observation.images.agentview"); the
policy was trained on the same names after the dataset wrapper's dot → underscore
renaming, so the mapping is just that renaming.
"""

from collections import deque

import numpy as np
import torch

from diffusion_policy.dataset.lerobot_wrapper import _safe_key


class DiffusionSimPolicy:
    def __init__(self, policy, shape_meta, device="cuda", n_action_steps=None):
        self.policy = policy.eval()
        self.device = device
        self.n_obs_steps = policy.n_obs_steps
        self.n_action_steps = n_action_steps or policy.n_action_steps
        self.obs_keys = list(shape_meta["obs"].keys())
        self.rgb_keys = {
            k for k, v in shape_meta["obs"].items() if v.get("type") == "rgb"
        }
        self._obs = deque(maxlen=self.n_obs_steps)
        self._actions = deque()

    def reset(self, seed=None):
        self._obs.clear()
        self._actions.clear()
        if seed is not None:
            # Diffusion sampling starts from torch.randn noise; seeding per episode
            # makes each episode reproducible on its own (same GPU / library versions)
            torch.manual_seed(seed)

    def _to_policy_obs(self, obs):
        out = {}
        for key, value in obs.items():
            k = _safe_key(key)
            if k not in self.obs_keys:
                continue
            if k in self.rgb_keys:
                # HWC uint8 → CHW float in [0, 1], as produced by the LeRobot loader
                value = np.moveaxis(value, -1, 0).astype(np.float32) / 255.0
            out[k] = np.asarray(value, dtype=np.float32)
        missing = set(self.obs_keys) - set(out)
        if missing:
            raise KeyError(f"Simulator obs lacks policy keys {sorted(missing)}")
        return out

    @torch.no_grad()
    def act(self, obs):
        frame = self._to_policy_obs(obs)
        if not self._obs:
            self._obs.extend([frame] * self.n_obs_steps)
        else:
            self._obs.append(frame)

        if not self._actions:
            batch = {
                k: torch.from_numpy(np.stack([o[k] for o in self._obs]))[None].to(
                    self.device
                )
                for k in self.obs_keys
            }
            chunk = self.policy.predict_action(batch)[0].cpu().numpy()
            self._actions.extend(chunk[: self.n_action_steps])
        return self._actions.popleft()
