# Stage 2 — how-to-code.md
# Sample temporal windows of length H from episode-based demonstration data.
# Handles padding at episode start/end boundaries.
# Reference: diffusion_policy-main/diffusion_policy/common/sampler.py

import numpy as np
from torch.utils.data import Dataset


class SequenceSampler(Dataset):
    def __init__(self, replay_buffer, sequence_length, pad_before=0, pad_after=0):
        # Build index: list of (episode_id, start_step) for all valid windows
        self.indices = []
        ...

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Returns dict of {key: [sequence_length, ...]} arrays
        # Pads with first/last frame at episode boundaries
        ...
