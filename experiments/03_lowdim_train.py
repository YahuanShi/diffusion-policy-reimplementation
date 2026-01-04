"""
Stage 3 validation — run this after implementing lowdim.py.

Trains the lowdim diffusion policy on PushT state data (no images).
Success criterion: training loss falls below 0.05 within ~100 epochs.
This isolates the diffusion logic from the vision encoder.
"""

import torch
from torch.utils.data import DataLoader

from diffusion_policy.model.diffusion.scheduler import DDPMScheduler
from diffusion_policy.model.diffusion.unet1d import ConditionalUnet1D
from diffusion_policy.dataset.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.sampler import SequenceSampler
from diffusion_policy.dataset.normalizer import LinearNormalizer
from diffusion_policy.policy.lowdim import DiffusionUnetLowdimPolicy

ZARR_PATH = 'data/pusht_cchi_v7_replay.zarr'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    # Dataset
    buf = ReplayBuffer(ZARR_PATH)
    sampler = SequenceSampler(buf, sequence_length=16, pad_before=1, pad_after=7)
    dataloader = DataLoader(sampler, batch_size=64, shuffle=True, num_workers=4)

    # Normalizer
    normalizer = LinearNormalizer()
    normalizer.fit({
        'obs':    torch.tensor(buf['state'][:], dtype=torch.float32),
        'action': torch.tensor(buf['action'][:], dtype=torch.float32),
    })

    # Policy
    model = ConditionalUnet1D(input_dim=2, global_cond_dim=2*5,
                               down_dims=[256, 512, 1024]).to(DEVICE)
    scheduler = DDPMScheduler(num_train_timesteps=100)
    policy = DiffusionUnetLowdimPolicy(
        model=model, noise_scheduler=scheduler,
        horizon=16, obs_dim=5, action_dim=2,
        n_obs_steps=2, n_action_steps=8,
    ).to(DEVICE)
    policy.set_normalizer(normalizer)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4, weight_decay=1e-6)

    for epoch in range(100):
        losses = []
        for batch in dataloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            loss = policy.compute_loss(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        print(f"epoch {epoch:>3}: loss={sum(losses)/len(losses):.4f}")


if __name__ == '__main__':
    main()
