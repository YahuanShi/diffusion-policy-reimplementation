"""
Evaluation entry point — loads a trained checkpoint and runs rollouts.

Loads a lowdim policy checkpoint, reconstructs the model architecture from
the saved normalizer dimensions, and evaluates over multiple episodes.

The model architecture (U-Net dims, horizon, etc.) is hardcoded here to match
the training config. In production, these should be saved in the checkpoint.

Usage:
    python eval.py --checkpoint outputs/policy_final.pt --mock   # mock env
    python eval.py --checkpoint outputs/policy_final.pt          # real env (needs robomimic)
"""

import argparse
import json
import torch

from diffusion_policy.model.diffusion.scheduler import DDPMScheduler
from diffusion_policy.model.diffusion.unet1d import ConditionalUnet1D
from diffusion_policy.dataset.normalizer import LinearNormalizer
from diffusion_policy.policy.lowdim import DiffusionUnetLowdimPolicy
from eval.runner import EvalRunner, MockEnv


def load_policy(checkpoint_path, device='cpu'):
    payload = torch.load(checkpoint_path, map_location=device)
    normalizer = LinearNormalizer()
    normalizer.load_state_dict(payload['normalizer_state_dict'])

    state_dict = payload['policy_state_dict']
    obs_dim = normalizer['obs'].scale.shape[0]
    action_dim = normalizer['action'].scale.shape[0]

    model = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * 2,
        down_dims=[256, 512, 1024],
        diffusion_step_embed_dim=256,
    )
    scheduler = DDPMScheduler(num_train_timesteps=100)
    policy = DiffusionUnetLowdimPolicy(
        model=model,
        noise_scheduler=scheduler,
        horizon=16,
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_obs_steps=2,
        n_action_steps=8,
        num_inference_steps=100,
    )
    policy.set_normalizer(normalizer)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--n_test', type=int, default=50)
    parser.add_argument('--mock', action='store_true')
    args = parser.parse_args()

    policy = load_policy(args.checkpoint, args.device)

    if args.mock:
        env_factory = lambda: MockEnv(
            obs_dim=policy.obs_dim,
            action_dim=policy.action_dim)
    else:
        raise NotImplementedError(
            "Real env evaluation requires robomimic. Use --mock for testing.")

    runner = EvalRunner(
        env_factory=env_factory,
        n_test=args.n_test,
        n_obs_steps=policy.n_obs_steps,
        n_action_steps=policy.n_action_steps,
    )
    result = runner.run(policy)
    print(f"mean_reward: {result['mean_reward']:.4f} +/- {result['std_reward']:.4f}")
    print(f"max_reward: {result['max_reward']:.4f}")
