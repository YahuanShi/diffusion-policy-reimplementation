# Stage 7 — how-to-code.md
# Vectorized evaluation: run N parallel episodes and compute mean_score.
# Reuses PushTImageEnv from the reference repo (not reimplemented).
# Reference: diffusion_policy-main/diffusion_policy/env_runner/pusht_image_runner.py

from collections import deque
import numpy as np
import torch

# Reuse the reference environment — only import from reference allowed
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv


def run_episode(env, policy, n_obs_steps, n_action_steps, max_steps=300):
    # 1. Reset env, initialise obs_buffer (deque of length n_obs_steps)
    # 2. Loop until done or max_steps:
    #      - build obs_dict with temporal stack
    #      - policy.predict_action(obs_dict) → action sequence
    #      - execute each action, update obs_buffer, track max reward
    # 3. Return max reward for episode
    ...


class PushTRunner:
    def __init__(self, n_test=50, max_steps=300, device='cuda'):
        self.n_test = n_test
        self.max_steps = max_steps
        self.device = device

    def run(self, policy):
        # Run n_test episodes, return {'mean_score': float, 'scores': list}
        ...
