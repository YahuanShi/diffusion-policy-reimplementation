"""
Training entry point for diffusion policy.

Image policy (LeRobot format):
    python train.py --repo_id lerobot/pusht --epochs 100
    python train.py --repo_id local/my_dataset --root /path/to/data --device cuda

Lowdim policy (legacy zarr format):
    python train.py --mode lowdim --data data/pick_place.zarr
"""

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train diffusion policy')
    parser.add_argument('--mode', default='image', choices=['image', 'lowdim'])

    # LeRobot (image mode)
    parser.add_argument('--repo_id', help='LeRobot dataset repo_id (e.g. lerobot/pusht)')
    parser.add_argument('--root', default=None, help='Local root for LeRobot dataset')
    parser.add_argument('--episodes', type=int, nargs='*', default=None)
    parser.add_argument('--state_key', default='observation.state')
    parser.add_argument('--image_keys', nargs='*', default=None)
    parser.add_argument('--resize', type=int, nargs=2, default=None, metavar=('H', 'W'),
                        help='Resize images to (H, W) before feeding to encoder')
    parser.add_argument('--num_workers', type=int, default=2)

    # Zarr (lowdim mode)
    parser.add_argument('--data', help='Path to zarr dataset (lowdim mode)')
    parser.add_argument('--obs_key', default='state')
    parser.add_argument('--action_key', default='action')

    # Shared
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--horizon', type=int, default=16)
    parser.add_argument('--n_obs_steps', type=int, default=2)
    parser.add_argument('--n_action_steps', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--checkpoint_every', type=int, default=100)

    # Wandb (enabled by default, use --no_wandb to disable)
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--wandb_run_name', default=None, help='Wandb run name')
    args = parser.parse_args()

    if args.mode == 'image':
        if not args.repo_id:
            parser.error('--repo_id is required for image mode')
        from training.workspace_image import train_image
        train_image(
            repo_id=args.repo_id,
            root=args.root,
            episodes=args.episodes,
            device=args.device,
            num_epochs=args.epochs,
            batch_size=args.batch,
            horizon=args.horizon,
            n_obs_steps=args.n_obs_steps,
            n_action_steps=args.n_action_steps,
            state_key=args.state_key,
            image_keys=args.image_keys,
            lr=args.lr,
            output_dir=args.output_dir,
            checkpoint_every=args.checkpoint_every,
            num_workers=args.num_workers,
            resize_shape=tuple(args.resize) if args.resize else None,
            use_wandb=not args.no_wandb,
            wandb_run_name=args.wandb_run_name,
        )
    else:
        if not args.data:
            parser.error('--data is required for lowdim mode')
        from training.workspace import train
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
            use_wandb=not args.no_wandb,
            wandb_run_name=args.wandb_run_name,
        )
