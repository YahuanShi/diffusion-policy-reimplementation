# Stage 2 — how-to-code.md
# Zarr-based replay buffer: reads demonstration data from disk.
# Reference: diffusion_policy-main/diffusion_policy/common/replay_buffer.py

import zarr
import numpy as np


class ReplayBuffer:
    def __init__(self, zarr_path):
        # Open the Zarr store and expose data/meta arrays
        ...

    @property
    def episode_ends(self):
        # Returns int64 array of cumulative step counts per episode
        ...

    def __getitem__(self, key):
        # Access data arrays by key: 'img', 'action', 'state'
        ...
