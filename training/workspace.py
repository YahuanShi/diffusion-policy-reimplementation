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
  Checkpoints saved every save_every_steps steps and every checkpoint_every epochs.

Usage:
  python train.py --repo_id lerobot/pusht --epochs 3000
  python train.py --repo_id local/my_data --root ./data --device cuda
"""

import glob
import json
import os
import random
import time
import torch
import wandb
import numpy as np
from torch.utils.data import DataLoader

from diffusion_policy.model.diffusion.scheduler import DDPMScheduler
from diffusion_policy.model.diffusion.ema import EMAModel
from diffusion_policy.model.vision.encoder import MultiImageObsEncoder
from diffusion_policy.dataset.lerobot_wrapper import LeRobotImageDataset
from diffusion_policy.policy.image import DiffusionUnetImagePolicy


def train_image(
    repo_id,
    root=None,
    episodes=None,
    device="cuda",
    batch_size=64,
    num_epochs=3000,
    horizon=16,
    n_obs_steps=2,
    n_action_steps=8,
    num_inference_steps=100,
    lr=1e-4,
    lr_warmup_steps=500,
    ema_power=2 / 3,
    checkpoint_every=100,
    save_every_steps=5000,
    max_keep_checkpoints=3,
    output_dir="outputs",
    image_keys=None,
    state_key="observation.state",
    use_group_norm=True,
    share_rgb_model=False,
    down_dims=(256, 512, 1024),
    diffusion_step_embed_dim=256,
    video_backend="pyav",
    num_workers=2,
    resize_shape=None,
    crop_shape=None,
    seed=42,
    wandb_run_name=None,
    resume_checkpoint=None,
):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.makedirs(output_dir, exist_ok=True)

    config = {
        "repo_id": repo_id,
        "root": root,
        "episodes": episodes,
        "device": device,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "horizon": horizon,
        "n_obs_steps": n_obs_steps,
        "n_action_steps": n_action_steps,
        "num_inference_steps": num_inference_steps,
        "lr": lr,
        "ema_power": ema_power,
        "checkpoint_every": checkpoint_every,
        "save_every_steps": save_every_steps,
        "image_keys": image_keys,
        "state_key": state_key,
        "down_dims": list(down_dims),
        "diffusion_step_embed_dim": diffusion_step_embed_dim,
        "num_workers": num_workers,
        "resize_shape": list(resize_shape) if resize_shape else None,
        "crop_shape": list(crop_shape) if crop_shape else None,
        "seed": seed,
        "max_keep_checkpoints": max_keep_checkpoints,
        "wandb_run_name": wandb_run_name,
        "resume_checkpoint": resume_checkpoint,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    dataset = LeRobotImageDataset(
        repo_id=repo_id,
        root=root,
        episodes=episodes,
        horizon=horizon,
        n_obs_steps=n_obs_steps,
        image_keys=image_keys,
        state_key=state_key,
        video_backend=video_backend,
    )
    shape_meta = dataset.shape_meta
    print(f"Dataset: {dataset.ds.num_episodes} episodes, {len(dataset)} frames")
    print(f"Shape meta: {shape_meta}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    print(f"Dataloader: {len(dataset)} samples, {len(dataloader)} batches/epoch")

    normalizer = dataset.get_normalizer()

    encoder = MultiImageObsEncoder(
        shape_meta,
        use_group_norm=use_group_norm,
        share_rgb_model=share_rgb_model,
        resize_shape=resize_shape,
        crop_shape=crop_shape,
        random_crop=(crop_shape is not None),
    )
    scheduler = DDPMScheduler(num_train_timesteps=100)
    policy = DiffusionUnetImagePolicy(
        obs_encoder=encoder,
        noise_scheduler=scheduler,
        shape_meta=shape_meta,
        horizon=horizon,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        num_inference_steps=num_inference_steps,
        diffusion_step_embed_dim=diffusion_step_embed_dim,
        down_dims=list(down_dims),
    )
    policy.set_normalizer(normalizer)
    policy.to(device)

    ema = EMAModel(policy, power=ema_power)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-6)
    total_steps = len(dataloader) * num_epochs
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=lr / 10
    )

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Model params: {n_params:,}")

    start_epoch = 0
    global_step = 0
    if resume_checkpoint is not None:
        ckpt = torch.load(resume_checkpoint, map_location=device)
        policy.load_state_dict(ckpt["policy_state_dict"])
        ema.load_state_dict(ckpt["ema_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        for _ in range(global_step):
            lr_scheduler.step()
        print(
            f"Resumed from {resume_checkpoint} (epoch {start_epoch}, step {global_step})"
        )

    print(f"Training for {num_epochs} epochs ({total_steps} steps)")

    wandb.init(
        project="Diffusion-Policy",
        name=wandb_run_name,
        config={
            "mode": "image",
            "repo_id": repo_id,
            "num_episodes": dataset.ds.num_episodes,
            "num_frames": len(dataset),
            "batch_size": batch_size,
            "num_epochs": num_epochs,
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

    for epoch in range(start_epoch, num_epochs):
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

            if save_every_steps > 0 and global_step % save_every_steps == 0:
                ckpt = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "policy_state_dict": policy.state_dict(),
                    "ema_state_dict": ema.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "normalizer_state_dict": normalizer.state_dict(),
                    "shape_meta": shape_meta,
                }
                path = os.path.join(output_dir, f"checkpoint_step{global_step}.pt")
                torch.save(ckpt, path)
                print(f"  saved {path}")
                if max_keep_checkpoints > 0:
                    step_ckpts = sorted(
                        glob.glob(os.path.join(output_dir, "checkpoint_step*.pt"))
                    )
                    for old in step_ckpts[:-max_keep_checkpoints]:
                        os.remove(old)

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
        if epoch % 50 == 0 or epoch == num_epochs - 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"epoch {epoch:>5}/{num_epochs}: loss={avg_loss:.4f} lr={lr_now:.2e} ema_decay={ema.decay:.4f} time={epoch_time:.1f}s"
            )

        if (epoch + 1) % checkpoint_every == 0 or epoch == num_epochs - 1:
            ckpt = {
                "epoch": epoch,
                "global_step": global_step,
                "policy_state_dict": policy.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "normalizer_state_dict": normalizer.state_dict(),
                "shape_meta": shape_meta,
            }
            path = os.path.join(output_dir, f"checkpoint_epoch{epoch + 1}.pt")
            torch.save(ckpt, path)
            print(f"  saved {path}")

    ema.copy_to(policy)
    final_path = os.path.join(output_dir, "policy_final.pt")
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "normalizer_state_dict": normalizer.state_dict(),
            "shape_meta": shape_meta,
        },
        final_path,
    )
    artifact = wandb.Artifact("policy", type="model", metadata=config)
    artifact.add_file(final_path)
    wandb.log_artifact(artifact)
    wandb.finish()
    print(f"Training complete. Final model: {final_path}")
    return policy
