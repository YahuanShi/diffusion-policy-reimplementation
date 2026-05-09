# Stage 2 — how-to-code.md
# Fit statistics from training data, normalize observations and actions to [-1, 1].
# Reference: diffusion_policy-main/diffusion_policy/model/common/normalizer.py

import torch


class SingleFieldNormalizer:
    def fit(self, data, mode='limits'):
        # mode='limits':   normalize to [-1, 1] using min/max
        # mode='gaussian': normalize to N(0, 1) using mean/std
        ...

    def normalize(self, x):
        # x: [..., D] → [..., D] in [-1, 1]
        ...

    def unnormalize(self, x):
        # Inverse of normalize
        ...

    def state_dict(self): ...
    def load_state_dict(self, d): ...


class LinearNormalizer:
    """Dict-based normalizer: normalizer['obs'].normalize(x)"""

    def fit(self, data_dict, mode='limits'):
        # data_dict: {key: tensor [N, D]}
        # Fits a SingleFieldNormalizer per key
        ...

    def __getitem__(self, key):
        ...

    def state_dict(self): ...
    def load_state_dict(self, d): ...
