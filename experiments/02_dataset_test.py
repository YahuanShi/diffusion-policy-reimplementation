"""
Stage 2 validation — run this after implementing replay_buffer.py,
sampler.py, and normalizer.py.

Loads the PushT Zarr dataset and verifies:
  - SequenceSampler produces windows of the correct shape
  - Episode boundaries are padded correctly (no cross-episode bleed)
  - LinearNormalizer maps data to [-1, 1]
"""

import torch
import numpy as np

from diffusion_policy.dataset.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.sampler import SequenceSampler
from diffusion_policy.dataset.normalizer import LinearNormalizer

ZARR_PATH = 'data/pusht_cchi_v7_replay.zarr'
HORIZON = 16
N_OBS_STEPS = 2
PAD_BEFORE = 1
PAD_AFTER = 7


def main():
    buf = ReplayBuffer(ZARR_PATH)
    print(f"Episodes: {len(buf.episode_ends)}")
    print(f"Total steps: {buf.episode_ends[-1]}")

    sampler = SequenceSampler(buf, sequence_length=HORIZON,
                               pad_before=PAD_BEFORE, pad_after=PAD_AFTER)
    print(f"Total samples: {len(sampler)}")

    sample = sampler[0]
    print("\nSample shapes:")
    for k, v in sample.items():
        print(f"  {k}: {v.shape}")

    # Verify no cross-episode bleed in first sample
    print("\nFirst 3 action steps of sample[0] (should be from episode 0 only):")
    print(sample['action'][:3])

    # Normalizer check
    actions = torch.tensor(buf['action'][:], dtype=torch.float32)
    normalizer = LinearNormalizer()
    normalizer.fit({'action': actions})
    normed = normalizer['action'].normalize(actions[:100])
    print(f"\nNormalized action range: [{normed.min():.3f}, {normed.max():.3f}]")
    print("Expected: close to [-1.0, 1.0]")


if __name__ == '__main__':
    main()
