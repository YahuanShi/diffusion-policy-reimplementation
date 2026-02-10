"""
Image policy training workspace — trains image-conditioned diffusion policy using LeRobot format.

Training uses the DDPM forward process (add noise) + MSE loss.
Inference uses DDIM (16 deterministic steps) — see inference.py / eval.py.
The trained weights are compatible with both; scheduler choice happens at load time.

Training loop:
  Each epoch:
    for batch in dataloader:
      1. Forward: policy.compute_loss(batch) → MSE(ε_θ, ε)  [DDPM forward process]
      2. Backward: loss.backward() + gradient clipping (max_norm=1.0)
      3. Update: optimizer.step() + lr_scheduler.step() + ema.step()
  Terminates after max_steps gradient steps. Checkpoints saved every save_every_steps steps;
  only the last max_keep_checkpoints are retained.

Validation (optional, --val_ratio > 0):
  A random subset of episodes is held out (split by episode, not by frame — neighbouring
  frames are nearly identical, so a frame-level split would leak). Every val_every_steps
  the EMA model is scored with evaluation.offline.evaluate_offline.

Checkpoint contents:
  policy_state_dict — RAW training weights (needed to resume optimisation)
  ema_state_dict    — EMA weights (used for inference; see diffusion_policy/policy/loader.py)
  config            — architecture + data settings, so loading never hardcodes dimensions

Usage:
  python train.py --repo_id local/my_data --root ./data --max_steps 30000
  python train.py --repo_id local/my_data --root ./data --max_steps 30000 --resume outputs/checkpoint_step20000.pt
"""

import glob
import json
import os
import random
import time

import numpy as np
import torch
import wandb
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from torch.utils.data import DataLoader

from diffusion_policy.dataset.lerobot_wrapper import LeRobotImageDataset
from diffusion_policy.model.diffusion.ema import EMAModel
from diffusion_policy.policy.loader import build_policy
from evaluation.offline import evaluate_offline


def split_episodes(repo_id, root, episodes, val_ratio, seed):
    """Split episode indices into (train, val) lists. val is empty when val_ratio == 0."""
    if episodes is None:
        n_total = LeRobotDatasetMetadata(repo_id, root=root).total_episodes
        episodes = list(range(n_total))
    episodes = sorted(episodes)
    if val_ratio <= 0:
        return episodes, []
    n_val = max(1, round(len(episodes) * val_ratio))
    if n_val >= len(episodes):
        raise ValueError(f"val_ratio={val_ratio} leaves no training episodes")
    rng = np.random.default_rng(seed)
    val = sorted(rng.choice(episodes, size=n_val, replace=False).tolist())
    val_set = set(val)
    train = [e for e in episodes if e not in val_set]
    return train, val


def _step_checkpoint_sort_key(path):
    # Numeric sort: lexicographic order would put step10000 before step5000
    return int(os.path.basename(path)[len("checkpoint_step") : -len(".pt")])


def train_image(
    repo_id,
    root=None,
    episodes=None,
    device="cuda",
    batch_size=64,
    max_steps=30000,
    horizon=16,
    n_obs_steps=2,
    n_action_steps=8,
    num_inference_steps=100,
    lr=1e-4,
    ema_power=2 / 3,
    save_every_steps=5000,
    max_keep_checkpoints=3,
    output_dir="outputs",
    image_keys=None,
    state_key="observation.state",
    use_group_norm=True,
    share_rgb_model=False,
    down_dims=(256, 512, 1024),
    diffusion_step_embed_dim=256,
    num_train_timesteps=100,
    video_backend="pyav",
    num_workers=8,
    resize_shape=None,
    crop_shape=None,
    val_ratio=0.0,
    val_every_steps=2000,
    max_val_batches=8,
    seed=42,
    wandb_run_name=None,
    resume_checkpoint=None,
):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.makedirs(output_dir, exist_ok=True)

    train_episodes, val_episodes = split_episodes(
        repo_id, root, episodes, val_ratio, seed
    )

    dataset = LeRobotImageDataset(
        repo_id=repo_id,
        root=root,
        episodes=train_episodes,
        horizon=horizon,
        n_obs_steps=n_obs_steps,
        image_keys=image_keys,
        state_key=state_key,
        video_backend=video_backend,
    )
    shape_meta = dataset.shape_meta
    print(f"Dataset: {dataset.ds.num_episodes} train episodes, {len(dataset)} frames")
    print(f"Shape meta: {shape_meta}")

    # Everything needed to rebuild the model + data pipeline later. Saved to config.json
    # AND inside every checkpoint, so a .pt file is self-describing.
    config = {
        "repo_id": repo_id,
        "root": root,
        "episodes": episodes,
        "train_episodes": train_episodes,
        "val_episodes": val_episodes,
        "device": device,
        "batch_size": batch_size,
        "max_steps": max_steps,
        "horizon": horizon,
        "n_obs_steps": n_obs_steps,
        "n_action_steps": n_action_steps,
        "num_inference_steps": num_inference_steps,
        "num_train_timesteps": num_train_timesteps,
        "lr": lr,
        "ema_power": ema_power,
        "save_every_steps": save_every_steps,
        "image_keys": dataset.image_keys,
        "state_key": state_key,
        "use_group_norm": use_group_norm,
        "share_rgb_model": share_rgb_model,
        "down_dims": list(down_dims),
        "diffusion_step_embed_dim": diffusion_step_embed_dim,
        "video_backend": video_backend,
        "num_workers": num_workers,
        "resize_shape": list(resize_shape) if resize_shape else None,
        "crop_shape": list(crop_shape) if crop_shape else None,
        "val_ratio": val_ratio,
        "val_every_steps": val_every_steps,
        "seed": seed,
        "max_keep_checkpoints": max_keep_checkpoints,
        "wandb_run_name": wandb_run_name,
        "resume_checkpoint": resume_checkpoint,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    print(f"Dataloader: {len(dataset)} samples, {len(dataloader)} batches/epoch")

    val_dataloader = None
    if val_episodes:
        val_dataset = LeRobotImageDataset(
            repo_id=repo_id,
            root=root,
            episodes=val_episodes,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            image_keys=dataset.image_keys,
            state_key=state_key,
            video_backend=video_backend,
        )
        # shuffle=True with a fixed generator: max_val_batches then covers all val
        # episodes (not just the first one) and the same samples every time.
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            num_workers=min(num_workers, 4),
            pin_memory=True,
        )
        print(f"Val: episodes {val_episodes}, {len(val_dataset)} frames")

    # Fit on training episodes only — val stays truly unseen
    normalizer = dataset.get_normalizer()

    policy = build_policy(shape_meta, config, num_inference_steps=num_inference_steps)
    policy.set_normalizer(normalizer)
    policy.to(device)

    ema = EMAModel(policy, power=ema_power)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-6)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_steps, eta_min=lr / 10
    )

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Model params: {n_params:,}")

    start_epoch = 0
    global_step = 0
    wandb_run_id = None
    if resume_checkpoint is not None:
        ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        # Raw weights go into the trained policy; EMA weights into the EMA copy.
        policy.load_state_dict(ckpt["policy_state_dict"])
        ema.load_state_dict(ckpt["ema_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        # Replay the schedule instead of loading its state: the saved state carries the
        # old T_max, which would be wrong if --max_steps changed between runs.
        for _ in range(global_step):
            lr_scheduler.step()
        wandb_run_id = ckpt.get("wandb_run_id")
        print(
            f"Resumed from {resume_checkpoint} (epoch {start_epoch}, step {global_step})"
        )

    remaining = max_steps - global_step
    print(f"Training for {max_steps} steps ({remaining} remaining)")

    # Passing the previous run id continues the same wandb run instead of starting a new one
    run = wandb.init(
        project="Diffusion-Policy",
        name=wandb_run_name,
        id=wandb_run_id,
        config={
            "mode": "image",
            "repo_id": repo_id,
            "num_episodes": dataset.ds.num_episodes,
            "num_val_episodes": len(val_episodes),
            "num_frames": len(dataset),
            "batch_size": batch_size,
            "max_steps": max_steps,
            "horizon": horizon,
            "n_obs_steps": n_obs_steps,
            "n_action_steps": n_action_steps,
            "lr": lr,
            "down_dims": list(down_dims),
            "resize_shape": resize_shape,
            "crop_shape": crop_shape,
            "n_params": n_params,
            "obs_keys": list(shape_meta["obs"].keys()),
            "action_dim": shape_meta["action"]["shape"][0],
            "seed": seed,
        },
        resume="allow",
    )

    def run_validation():
        metrics = evaluate_offline(
            ema.averaged_model, val_dataloader, device, max_batches=max_val_batches
        )
        log = {
            "val/loss": metrics["loss"],
            "val/action_mse": metrics["action_mse"],
            "val/action_mae": metrics["action_mae"],
        }
        for d, v in enumerate(metrics["action_mae_per_dim"]):
            log[f"val/action_mae_dim{d}"] = v
        wandb.log(log, step=global_step)
        print(
            f"  val step {global_step}: loss={metrics['loss']:.4f} "
            f"action_mse={metrics['action_mse']:.4f} action_mae={metrics['action_mae']:.4f}"
        )

    done = global_step >= max_steps
    epoch = start_epoch
    while not done:
        policy.train()
        epoch_losses = []
        epoch_start = time.time()

        for batch in dataloader:
            batch_gpu = {k: v.to(device) for k, v in batch.items()}
            loss = policy.compute_loss(batch_gpu)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            ema.step(policy)

            step_loss = loss.item()
            epoch_losses.append(step_loss)
            global_step += 1

            wandb.log(
                {
                    "train/loss": step_loss,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/ema_decay": ema.decay,
                    "train/grad_norm": grad_norm.item(),
                },
                step=global_step,
            )

            if val_dataloader is not None and global_step % val_every_steps == 0:
                run_validation()

            if save_every_steps > 0 and global_step % save_every_steps == 0:
                ckpt = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "policy_state_dict": policy.state_dict(),
                    "ema_state_dict": ema.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "normalizer_state_dict": normalizer.state_dict(),
                    "shape_meta": shape_meta,
                    "config": config,
                    "wandb_run_id": run.id,
                }
                path = os.path.join(output_dir, f"checkpoint_step{global_step}.pt")
                torch.save(ckpt, path)
                print(f"  saved {path}")
                if max_keep_checkpoints > 0:
                    step_ckpts = sorted(
                        glob.glob(os.path.join(output_dir, "checkpoint_step*.pt")),
                        key=_step_checkpoint_sort_key,
                    )
                    for old in step_ckpts[:-max_keep_checkpoints]:
                        os.remove(old)

            if global_step >= max_steps:
                done = True
                break

        epoch_time = time.time() - epoch_start
        avg_loss = np.mean(epoch_losses)
        epoch_log = {
            "epoch/loss": avg_loss,
            "epoch/epoch": epoch,
            "epoch/time_sec": epoch_time,
        }
        if torch.cuda.is_available():
            epoch_log["epoch/gpu_mem_gb"] = (
                torch.cuda.max_memory_allocated(device) / 1e9
            )
            torch.cuda.reset_peak_memory_stats(device)
        wandb.log(epoch_log, step=global_step)
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch:>5} step {global_step:>6}/{max_steps}: loss={avg_loss:.4f} lr={lr_now:.2e} ema_decay={ema.decay:.4f} time={epoch_time:.1f}s"
        )

        epoch += 1

    if val_dataloader is not None and global_step % val_every_steps != 0:
        run_validation()

    ema.copy_to(policy)
    final_path = os.path.join(output_dir, "policy_final.pt")
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "normalizer_state_dict": normalizer.state_dict(),
            "shape_meta": shape_meta,
            "config": config,
        },
        final_path,
    )
    artifact = wandb.Artifact("policy", type="model", metadata=config)
    artifact.add_file(final_path)
    wandb.log_artifact(artifact)
    wandb.finish()
    print(f"Training complete. Final model: {final_path}")
    return policy
