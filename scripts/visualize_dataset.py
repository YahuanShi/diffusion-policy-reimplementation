"""
Dataset statistics visualization — quick sanity check for LeRobot datasets.

Outputs:
  - Action dimension distributions (histogram per dim)
  - Episode length distribution
  - Sample action trajectory from first episode

Usage:
    python scripts/visualize_dataset.py --repo_id local/ur5_pick_place --root ./data
    python scripts/visualize_dataset.py --repo_id lerobot/pusht --output_dir ./plots
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def plot_action_distributions(actions: torch.Tensor, output_dir: str):
    actions_np = actions.numpy()
    n_dims = actions_np.shape[1]

    cols = min(4, n_dims)
    rows = (n_dims + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.array(axes).flatten() if n_dims > 1 else [axes]

    for i in range(n_dims):
        axes[i].hist(actions_np[:, i], bins=50, color="steelblue", alpha=0.8)
        axes[i].set_title(f"action dim {i}")
        axes[i].set_xlabel("value")
        axes[i].set_ylabel("count")
        axes[i].axvline(
            actions_np[:, i].mean(),
            color="red",
            linestyle="--",
            linewidth=1,
            label="mean",
        )
        axes[i].legend(fontsize=8)

    for i in range(n_dims, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Action dimension distributions", fontsize=14)
    fig.tight_layout()
    path = os.path.join(output_dir, "action_distributions.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved {path}")


def episode_bounds(ds: LeRobotDataset):
    """[(from, to), ...] frame index range of each episode (LeRobot v3 metadata)."""
    eps = ds.meta.episodes
    return list(zip(eps["dataset_from_index"], eps["dataset_to_index"]))


def plot_episode_lengths(ds: LeRobotDataset, output_dir: str):
    lengths = [to - frm for frm, to in episode_bounds(ds)]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lengths, bins=30, color="darkorange", alpha=0.8)
    ax.set_xlabel("Episode length (frames)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Episode length distribution  (n={len(lengths)}, mean={np.mean(lengths):.0f})"
    )
    ax.axvline(np.mean(lengths), color="red", linestyle="--", linewidth=1, label="mean")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(output_dir, "episode_lengths.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved {path}")


def plot_sample_trajectory(ds: LeRobotDataset, actions: torch.Tensor, output_dir: str):
    ep_start, ep_end = episode_bounds(ds)[0]
    actions_np = actions[ep_start:ep_end].numpy()
    n_dims = actions_np.shape[1]
    T = actions_np.shape[0]

    fig, ax = plt.subplots(figsize=(12, 4))
    for i in range(n_dims):
        ax.plot(range(T), actions_np[:, i], label=f"dim {i}", alpha=0.8)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Action value")
    ax.set_title("Action trajectory — episode 0")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = os.path.join(output_dir, "sample_trajectory.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--action_key", default="action")
    parser.add_argument("--output_dir", default="plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading dataset: {args.repo_id} ...")
    ds = LeRobotDataset(args.repo_id, root=args.root)
    print(f"  episodes: {ds.num_episodes}  frames: {len(ds)}")

    # Column access is much faster than row-by-row iteration
    actions = torch.from_numpy(np.array(ds.hf_dataset[args.action_key]))
    print(
        f"  action shape: {actions.shape}  min: {actions.min():.3f}  max: {actions.max():.3f}"
    )

    print("Generating plots ...")
    plot_action_distributions(actions, args.output_dir)
    plot_episode_lengths(ds, args.output_dir)
    plot_sample_trajectory(ds, actions, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
