"""
Full training entry point — run after completing all stages.
See how-to-code.md Stage 8 for integration details.

Usage:
    uv run python train.py
    uv run python train.py --device cpu --epochs 100   # quick smoke test
"""

import argparse
from training.workspace import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',    default='data/pusht_cchi_v7_replay.zarr')
    parser.add_argument('--device',  default='cuda')
    parser.add_argument('--epochs',  type=int, default=3050)
    parser.add_argument('--batch',   type=int, default=64)
    args = parser.parse_args()

    train(
        zarr_path=args.data,
        device=args.device,
        num_epochs=args.epochs,
        batch_size=args.batch,
        horizon=16,
        n_obs_steps=2,
        n_action_steps=8,
    )
