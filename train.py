"""
Training entry point for diffusion policy (image mode, LeRobot format).

Usage:
    python train.py --repo_id local/my_dataset --root ./data/my_dataset --epochs 3000
    python train.py --repo_id local/my_dataset --root ./data/my_dataset --resume outputs/checkpoint_epoch500.pt
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Train diffusion policy")

    # Dataset
    parser.add_argument("--repo_id", required=True, help="LeRobot dataset repo_id")
    parser.add_argument("--root", default=None, help="Local root for LeRobot dataset")
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--state_key", default="observation.state")
    parser.add_argument("--image_keys", nargs="*", default=None)
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Resize images to (H, W) before cropping",
    )
    parser.add_argument(
        "--crop",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Crop images to (H, W) — random in training, center at eval",
    )
    parser.add_argument("--num_workers", type=int, default=2)

    # Training
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n_obs_steps", type=int, default=2)
    parser.add_argument("--n_action_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--checkpoint_every", type=int, default=100)
    parser.add_argument(
        "--save_every_steps",
        type=int,
        default=5000,
        help="Save a checkpoint every N gradient steps (0 to disable)",
    )
    parser.add_argument(
        "--max_keep_checkpoints",
        type=int,
        default=3,
        help="Max step-checkpoints to keep (0 to keep all)",
    )
    parser.add_argument(
        "--resume", default=None, help="Path to checkpoint to resume training from"
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    # Wandb
    parser.add_argument("--wandb_run_name", default=None, help="Wandb run name")
    args = parser.parse_args()

    from training.workspace import train_image

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
        save_every_steps=args.save_every_steps,
        max_keep_checkpoints=args.max_keep_checkpoints,
        num_workers=args.num_workers,
        resize_shape=tuple(args.resize) if args.resize else None,
        crop_shape=tuple(args.crop) if args.crop else None,
        seed=args.seed,
        wandb_run_name=args.wandb_run_name,
        resume_checkpoint=args.resume,
    )


if __name__ == "__main__":
    main()
