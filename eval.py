"""
Evaluation entry point — loads a trained image policy checkpoint and runs rollouts.

The checkpoint contains shape_meta and normalizer, so the model architecture
is fully reconstructed without hardcoding dimensions.

Usage:
    python eval.py --checkpoint outputs/policy_final.pt --mock
    python eval.py --checkpoint outputs/policy_final.pt --device cuda --mock
"""

import argparse
import torch

from diffusion_policy.model.diffusion.scheduler import DDIMScheduler
from diffusion_policy.model.vision.encoder import MultiImageObsEncoder
from diffusion_policy.dataset.normalizer import LinearNormalizer
from diffusion_policy.policy.image import DiffusionUnetImagePolicy
from evaluation.runner import EvalRunner, MockEnv


def load_policy(checkpoint_path, device='cpu', resize_shape=None):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    shape_meta = payload['shape_meta']

    normalizer = LinearNormalizer()
    normalizer.load_state_dict(payload['normalizer_state_dict'])

    encoder = MultiImageObsEncoder(
        shape_meta, use_group_norm=True,
        resize_shape=resize_shape,
    )
    scheduler = DDIMScheduler(num_train_timesteps=100)
    policy = DiffusionUnetImagePolicy(
        obs_encoder=encoder,
        noise_scheduler=scheduler,
        shape_meta=shape_meta,
        horizon=16,
        n_obs_steps=2,
        n_action_steps=8,
        num_inference_steps=16,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
    )
    policy.set_normalizer(normalizer)
    policy.load_state_dict(payload['policy_state_dict'])
    policy.to(device)
    policy.eval()
    return policy, shape_meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--n_test', type=int, default=50)
    parser.add_argument('--mock', action='store_true')
    parser.add_argument('--resize', type=int, nargs=2, default=None, metavar=('H', 'W'))
    args = parser.parse_args()

    policy, shape_meta = load_policy(
        args.checkpoint, args.device,
        resize_shape=tuple(args.resize) if args.resize else None)

    if args.mock:
        def env_factory():
            return MockEnv(shape_meta=shape_meta)
    else:
        raise NotImplementedError(
            "Real env evaluation requires a gym environment. Use --mock for testing.")

    runner = EvalRunner(
        env_factory=env_factory,
        shape_meta=shape_meta,
        n_test=args.n_test,
        n_obs_steps=2,
        n_action_steps=8,
    )
    result = runner.run(policy)
    print(f"mean_reward: {result['mean_reward']:.4f} +/- {result['std_reward']:.4f}")
    print(f"max_reward: {result['max_reward']:.4f}")
