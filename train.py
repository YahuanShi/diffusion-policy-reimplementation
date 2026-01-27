"""
Full training entry point for Pick and Place.

Usage:
    python train.py --data data/pick_place.zarr
    python train.py --data data/pick_place.zarr --device cpu --epochs 100
"""

import argparse
from training.workspace import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train diffusion policy')
    parser.add_argument('--data', required=True, help='Path to zarr dataset')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--horizon', type=int, default=16)
    parser.add_argument('--n_obs_steps', type=int, default=2)
    parser.add_argument('--n_action_steps', type=int, default=8)
    parser.add_argument('--obs_key', default='state')
    parser.add_argument('--action_key', default='action')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--checkpoint_every', type=int, default=100)
    args = parser.parse_args()

    train(
        zarr_path=args.data,
        device=args.device,
        num_epochs=args.epochs,
        batch_size=args.batch,
        horizon=args.horizon,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
        obs_key=args.obs_key,
        action_key=args.action_key,
        lr=args.lr,
        output_dir=args.output_dir,
        checkpoint_every=args.checkpoint_every,
    )
