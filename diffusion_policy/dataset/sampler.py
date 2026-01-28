"""
Sequence sampler — extracts fixed-length windows from episode data for training.

The sampler slides a window of `sequence_length` steps across each episode.
Padding (pad_before, pad_after) allows the window to extend beyond episode boundaries,
using edge-value replication (first/last frame repeated).

Why padding matters for Diffusion Policy:
  - pad_before = n_obs_steps - 1: ensures every timestep has enough observation history
  - pad_after = n_action_steps - 1: ensures every timestep has a full action prediction horizon
  - Without padding, the first/last few frames of each episode would be unusable

Each window yields a dict of {key: (sequence_length, *data_shape)} arrays.
"""

import numpy as np
from torch.utils.data import Dataset


class SequenceSampler(Dataset):
    def __init__(self, replay_buffer, sequence_length, pad_before=0, pad_after=0, keys=None):
        self.replay_buffer = replay_buffer
        self.sequence_length = sequence_length
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.keys = keys or replay_buffer.keys()

        episode_ends = replay_buffer.episode_ends
        episode_starts = np.concatenate([[0], episode_ends[:-1]])

        # Build valid (buffer_start, buffer_end, sample_start, sample_end) tuples
        indices = []
        for ep_start, ep_end in zip(episode_starts, episode_ends):
            ep_len = ep_end - ep_start
            # With padding, we can start pad_before steps before the episode
            # and end pad_after steps after the episode
            n_windows = ep_len + pad_before + pad_after - sequence_length + 1
            for offset in range(max(n_windows, 0)):
                # offset is relative to (ep_start - pad_before)
                global_start = ep_start - pad_before + offset
                global_end = global_start + sequence_length

                # Clamp to actual episode bounds
                buf_start = max(global_start, ep_start)
                buf_end = min(global_end, ep_end)

                # Where in the output window does the buffer data land
                sample_start = buf_start - global_start
                sample_end = sample_start + (buf_end - buf_start)

                indices.append([buf_start, buf_end, sample_start, sample_end])

        self.indices = np.array(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        buf_start, buf_end, sample_start, sample_end = self.indices[idx]
        result = {}
        for key in self.keys:
            data = np.array(self.replay_buffer[key][buf_start:buf_end])
            # Allocate output and pad with edge values
            shape = (self.sequence_length,) + data.shape[1:]
            out = np.zeros(shape, dtype=data.dtype)
            out[sample_start:sample_end] = data
            # Pad before with first frame
            if sample_start > 0:
                out[:sample_start] = data[0]
            # Pad after with last frame
            if sample_end < self.sequence_length:
                out[sample_end:] = data[-1]
            result[key] = out
        return result


if __name__ == '__main__':
    import zarr, tempfile, os

    # Create a fake zarr dataset for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'test.zarr')
        root = zarr.open(path, mode='w')
        data = root.create_group('data')
        meta = root.create_group('meta')

        # 2 episodes: 10 steps, 5 steps
        action = np.arange(30).reshape(15, 2).astype(np.float32)
        data.create_dataset('action', data=action)
        meta.create_dataset('episode_ends', data=np.array([10, 15]))

        from diffusion_policy.dataset.replay_buffer import ReplayBuffer
        buf = ReplayBuffer(path)
        sampler = SequenceSampler(buf, sequence_length=4, pad_before=1, pad_after=1, keys=['action'])

        print(f"num_windows: {len(sampler)}")
        sample = sampler[0]
        print(f"sample[0] action shape: {sample['action'].shape}")
        print(f"sample[0] action:\n{sample['action']}")

        # Check padding: first sample should pad before with first frame
        assert sample['action'].shape == (4, 2)
        # First row should be padded (same as second row)
        assert np.allclose(sample['action'][0], sample['action'][1]), "pad_before should repeat first frame"
        print("SequenceSampler OK")
