"""
Stage 2 validation — tests replay_buffer.py, sampler.py, normalizer.py.

Uses a synthetic zarr dataset (no external data required).
Verifies: correct window shapes, episode padding, normalizer range.
"""

import torch
import numpy as np
import zarr
import tempfile
import os

from diffusion_policy.dataset.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.sampler import SequenceSampler
from diffusion_policy.dataset.normalizer import LinearNormalizer

HORIZON = 16
PAD_BEFORE = 1
PAD_AFTER = 7


def create_synthetic_zarr(path, n_episodes=5, ep_lengths=None):
    if ep_lengths is None:
        ep_lengths = [20, 15, 25, 10, 30]
    root = zarr.open(path, mode='w')
    data = root.create_group('data')
    meta = root.create_group('meta')

    total = sum(ep_lengths)
    action = np.random.randn(total, 2).astype(np.float32)
    state = np.random.randn(total, 4).astype(np.float32)
    data.create_dataset('action', data=action)
    data.create_dataset('state', data=state)
    meta.create_dataset('episode_ends', data=np.cumsum(ep_lengths))
    return path


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = os.path.join(tmpdir, 'test.zarr')
        create_synthetic_zarr(zarr_path)

        buf = ReplayBuffer(zarr_path)
        print(f"Episodes: {buf.n_episodes}")
        print(f"Total steps: {buf.episode_ends[-1]}")
        print(f"Episode lengths: {buf.episode_lengths}")

        sampler = SequenceSampler(buf, sequence_length=HORIZON,
                                   pad_before=PAD_BEFORE, pad_after=PAD_AFTER,
                                   keys=['action', 'state'])
        print(f"Total windows: {len(sampler)}")
        assert len(sampler) > 0

        sample = sampler[0]
        print(f"\nSample shapes:")
        for k, v in sample.items():
            print(f"  {k}: {v.shape}")
            assert v.shape[0] == HORIZON

        # Verify padding: first sample should have pad_before replicated frames
        s0 = sampler[0]
        assert np.allclose(s0['action'][0], s0['action'][1]), \
            "First padded frame should match first real frame"

        # Get last sample of first episode to check pad_after
        ep0_len = buf.episode_lengths[0]
        n_windows_ep0 = ep0_len + PAD_BEFORE + PAD_AFTER - HORIZON + 1
        last_ep0 = sampler[n_windows_ep0 - 1]
        assert np.allclose(last_ep0['action'][-1], last_ep0['action'][-2]), \
            "Last padded frame should match last real frame"

        # Cross-episode: consecutive episodes should not bleed
        ep0_data = buf.get_episode(0)
        ep1_data = buf.get_episode(1)
        ep0_last_action = ep0_data['action'][-1]
        ep1_first_action = ep1_data['action'][0]
        for i in range(len(sampler)):
            s = sampler[i]
            if np.allclose(s['action'][HORIZON // 2], ep0_last_action):
                next_frame = s['action'][HORIZON // 2 + 1]
                assert not np.allclose(next_frame, ep1_first_action) or \
                    np.allclose(next_frame, ep0_last_action), \
                    "Cross-episode bleed detected!"
                break

        # Normalizer
        actions = torch.tensor(buf['action'][:], dtype=torch.float32)
        normalizer = LinearNormalizer()
        normalizer.fit({'action': actions, 'state': torch.tensor(buf['state'][:], dtype=torch.float32)})
        normed = normalizer['action'].normalize(actions)
        print(f"\nNormalized action range: [{normed.min():.3f}, {normed.max():.3f}]")
        assert normed.min() >= -1.01 and normed.max() <= 1.01

        recovered = normalizer['action'].unnormalize(normed)
        assert torch.allclose(recovered, actions, atol=1e-5)

        # State dict round-trip
        sd = normalizer.state_dict()
        n2 = LinearNormalizer()
        n2.load_state_dict(sd)
        normed2 = n2['action'].normalize(actions)
        assert torch.allclose(normed, normed2)

    print("\nStage 2 validation PASSED")


if __name__ == '__main__':
    main()
