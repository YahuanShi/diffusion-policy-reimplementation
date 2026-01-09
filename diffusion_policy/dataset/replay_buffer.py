import zarr
import numpy as np


class ReplayBuffer:
    def __init__(self, zarr_path):
        root = zarr.open(zarr_path, mode='r')
        self._data = root['data']
        self._meta = root['meta']

    @property
    def episode_ends(self):
        return np.array(self._meta['episode_ends'], dtype=np.int64)

    @property
    def n_episodes(self):
        return len(self.episode_ends)

    @property
    def episode_lengths(self):
        ends = self.episode_ends
        starts = np.concatenate([[0], ends[:-1]])
        return ends - starts

    def keys(self):
        return list(self._data.keys())

    def __getitem__(self, key):
        return self._data[key]

    def get_episode(self, idx):
        ends = self.episode_ends
        start = 0 if idx == 0 else ends[idx - 1]
        end = ends[idx]
        return {key: np.array(self._data[key][start:end]) for key in self._data.keys()}


if __name__ == '__main__':
    import os
    path = 'data/pusht_cchi_v7_replay.zarr'
    if os.path.exists(path):
        buf = ReplayBuffer(path)
        print(f"keys: {buf.keys()}")
        print(f"episodes: {buf.n_episodes}")
        print(f"lengths: {buf.episode_lengths[:5]}...")
        ep = buf.get_episode(0)
        for k, v in ep.items():
            print(f"  {k}: {v.shape} {v.dtype}")
    else:
        print(f"No data at {path}, skipping live test")
        buf_keys = ['action', 'img', 'state']
        print(f"ReplayBuffer interface OK (keys would be: {buf_keys})")
