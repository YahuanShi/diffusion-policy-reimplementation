import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from diffusion_policy.model.diffusion.scheduler import DDPMScheduler
from diffusion_policy.model.diffusion.unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.ema import EMAModel
from diffusion_policy.dataset.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.sampler import SequenceSampler
from diffusion_policy.dataset.normalizer import LinearNormalizer
from diffusion_policy.policy.lowdim import DiffusionUnetLowdimPolicy


def train(zarr_path, device='cuda', batch_size=64, num_epochs=3000,
          horizon=16, n_obs_steps=2, n_action_steps=8,
          obs_key='state', action_key='action',
          lr=1e-4, lr_warmup_steps=500,
          ema_power=2/3, checkpoint_every=100,
          output_dir='outputs'):

    os.makedirs(output_dir, exist_ok=True)

    buf = ReplayBuffer(zarr_path)
    obs_dim = buf[obs_key].shape[-1]
    action_dim = buf[action_key].shape[-1]
    print(f"Dataset: {buf.n_episodes} episodes, obs_dim={obs_dim}, action_dim={action_dim}")

    sampler = SequenceSampler(
        buf, sequence_length=horizon,
        pad_before=n_obs_steps - 1,
        pad_after=n_action_steps - 1,
        keys=[obs_key, action_key])
    def collate_fn(batch):
        result = {}
        for key in batch[0]:
            arrays = [b[key] for b in batch]
            result[key] = torch.tensor(np.stack(arrays), dtype=torch.float32)
        return result

    dataloader = DataLoader(sampler, batch_size=batch_size, shuffle=True,
                            num_workers=0, collate_fn=collate_fn)
    print(f"Dataloader: {len(sampler)} windows, {len(dataloader)} batches/epoch")

    normalizer = LinearNormalizer()
    normalizer.fit({
        'obs': torch.tensor(buf[obs_key][:], dtype=torch.float32),
        'action': torch.tensor(buf[action_key][:], dtype=torch.float32),
    })

    model = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * n_obs_steps,
        down_dims=[256, 512, 1024],
        diffusion_step_embed_dim=256,
    )
    scheduler = DDPMScheduler(num_train_timesteps=100)
    policy = DiffusionUnetLowdimPolicy(
        model=model,
        noise_scheduler=scheduler,
        horizon=horizon,
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        num_inference_steps=100,
    )
    policy.set_normalizer(normalizer)
    policy.to(device)

    ema = EMAModel(policy, power=ema_power)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-6)
    total_steps = len(dataloader) * num_epochs
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=lr / 10)

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Model params: {n_params:,}")
    print(f"Training for {num_epochs} epochs ({total_steps} steps)")

    global_step = 0
    for epoch in range(num_epochs):
        policy.train()
        epoch_losses = []

        for batch in dataloader:
            batch_dict = {
                'obs': batch[obs_key].to(device),
                'action': batch[action_key].to(device),
            }
            loss = policy.compute_loss(batch_dict)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            ema.step(policy)

            epoch_losses.append(loss.item())
            global_step += 1

        avg_loss = np.mean(epoch_losses)
        if epoch % 50 == 0 or epoch == num_epochs - 1:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"epoch {epoch:>5}/{num_epochs}: loss={avg_loss:.4f} lr={lr_now:.2e} ema_decay={ema.decay:.4f}")

        if (epoch + 1) % checkpoint_every == 0 or epoch == num_epochs - 1:
            ckpt = {
                'epoch': epoch,
                'global_step': global_step,
                'policy_state_dict': policy.state_dict(),
                'ema_state_dict': ema.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'normalizer_state_dict': normalizer.state_dict(),
            }
            path = os.path.join(output_dir, f'checkpoint_epoch{epoch+1}.pt')
            torch.save(ckpt, path)
            print(f"  saved {path}")

    # Save final EMA weights
    ema.copy_to(policy)
    final_path = os.path.join(output_dir, 'policy_final.pt')
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'normalizer_state_dict': normalizer.state_dict(),
    }, final_path)
    print(f"Training complete. Final model: {final_path}")
    return policy
