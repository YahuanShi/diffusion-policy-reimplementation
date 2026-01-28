"""
Evaluation runner — rollout a trained policy in an environment and collect metrics.

Execution loop (mirrors real robot deployment):
  1. Maintain an observation deque of the last n_obs_steps frames
  2. Stack observations → policy.predict_action() → get n_action_steps actions
  3. Execute actions one by one in the environment, collecting rewards
  4. Repeat until done or max_steps reached

The action chunking pattern (predict n_action_steps, execute all, then re-plan)
is a key Diffusion Policy design: it provides temporal consistency and reduces
the compounding error of single-step prediction.
"""

from collections import deque
import numpy as np
import torch


def run_episode(env, policy, n_obs_steps, n_action_steps, max_steps=400):
    obs = env.reset()
    obs_deque = deque([obs] * n_obs_steps, maxlen=n_obs_steps)
    total_reward = 0.0
    done = False
    step = 0

    policy.eval()
    while not done and step < max_steps:
        obs_stack = np.stack(list(obs_deque), axis=0)
        obs_tensor = torch.tensor(obs_stack, dtype=torch.float32).unsqueeze(0)
        obs_dict = {'obs': obs_tensor}

        with torch.no_grad():
            action_seq = policy.predict_action(obs_dict)

        action_seq = action_seq[0].cpu().numpy()

        for a_idx in range(min(n_action_steps, len(action_seq))):
            if done or step >= max_steps:
                break
            obs, reward, done, info = env.step(action_seq[a_idx])
            obs_deque.append(obs)
            total_reward += reward
            step += 1

    return {'total_reward': total_reward, 'steps': step, 'done': done}


class EvalRunner:
    def __init__(self, env_factory, n_test=50, max_steps=400,
                 n_obs_steps=2, n_action_steps=8):
        self.env_factory = env_factory
        self.n_test = n_test
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps

    def run(self, policy):
        results = []
        for i in range(self.n_test):
            env = self.env_factory()
            result = run_episode(
                env, policy,
                n_obs_steps=self.n_obs_steps,
                n_action_steps=self.n_action_steps,
                max_steps=self.max_steps)
            results.append(result)

        rewards = [r['total_reward'] for r in results]
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'n_episodes': len(results),
            'results': results,
        }


class MockEnv:
    def __init__(self, obs_dim=4, action_dim=2, episode_len=50):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.episode_len = episode_len
        self._step = 0

    def reset(self):
        self._step = 0
        return np.zeros(self.obs_dim, dtype=np.float32)

    def step(self, action):
        self._step += 1
        obs = np.random.randn(self.obs_dim).astype(np.float32) * 0.1
        reward = -np.sum(action ** 2)
        done = self._step >= self.episode_len
        return obs, reward, done, {}
