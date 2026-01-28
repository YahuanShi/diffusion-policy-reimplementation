"""
Data normalizer — maps state/action to a uniform range for easier network learning.

Two modes:
  limits (default): linear map to [-1, 1]
    scale = 2 / (max - min),  offset = -1 - scale * min
  gaussian: standardize to mean=0, std=1
    scale = 1/std,  offset = -mean/std

normalize / unnormalize:
  normalize:   y = x * scale + offset
  unnormalize: x = (y - offset) / scale

Critical for Diffusion Policy:
  - DDPM's clip operation assumes data in [-1, 1]
  - State normalization unifies dimensions with different units (e.g. joint angles in rad vs position in m)
"""

import torch


class SingleFieldNormalizer:
    def __init__(self):
        self.scale = None
        self.offset = None

    def fit(self, data, mode='limits', output_min=-1.0, output_max=1.0, eps=1e-6):
        if isinstance(data, torch.Tensor):
            data = data.float()
        else:
            data = torch.tensor(data, dtype=torch.float32)

        flat = data.reshape(-1, data.shape[-1])

        if mode == 'limits':
            input_min = flat.min(dim=0).values
            input_max = flat.max(dim=0).values
            input_range = input_max - input_min
            input_range = torch.clamp(input_range, min=eps)
            scale = (output_max - output_min) / input_range
            offset = output_min - scale * input_min
        elif mode == 'gaussian':
            mean = flat.mean(dim=0)
            std = flat.std(dim=0)
            std = torch.clamp(std, min=eps)
            scale = 1.0 / std
            offset = -mean / std
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.scale = scale
        self.offset = offset
        return self

    def normalize(self, x):
        scale = self.scale.to(x.device)
        offset = self.offset.to(x.device)
        return x * scale + offset

    def unnormalize(self, x):
        scale = self.scale.to(x.device)
        offset = self.offset.to(x.device)
        return (x - offset) / scale

    def state_dict(self):
        return {'scale': self.scale, 'offset': self.offset}

    def load_state_dict(self, d):
        self.scale = d['scale']
        self.offset = d['offset']


class LinearNormalizer:
    def __init__(self):
        self._normalizers = {}

    def fit(self, data_dict, mode='limits', **kwargs):
        for key, data in data_dict.items():
            n = SingleFieldNormalizer()
            n.fit(data, mode=mode, **kwargs)
            self._normalizers[key] = n
        return self

    def __getitem__(self, key):
        return self._normalizers[key]

    def __contains__(self, key):
        return key in self._normalizers

    def keys(self):
        return self._normalizers.keys()

    def state_dict(self):
        return {k: v.state_dict() for k, v in self._normalizers.items()}

    def load_state_dict(self, d):
        for k, sd in d.items():
            n = SingleFieldNormalizer()
            n.load_state_dict(sd)
            self._normalizers[k] = n


if __name__ == '__main__':
    # Test SingleFieldNormalizer
    data = torch.tensor([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]])
    n = SingleFieldNormalizer()
    n.fit(data, mode='limits')
    normed = n.normalize(data)
    print(f"limits: min={normed.min().item():.1f}, max={normed.max().item():.1f}")
    assert torch.allclose(normed.min(dim=0).values, torch.tensor([-1.0, -1.0]))
    assert torch.allclose(normed.max(dim=0).values, torch.tensor([1.0, 1.0]))

    recovered = n.unnormalize(normed)
    assert torch.allclose(recovered, data)

    # Test gaussian mode
    n2 = SingleFieldNormalizer()
    n2.fit(data, mode='gaussian')
    normed2 = n2.normalize(data)
    print(f"gaussian: mean={normed2.mean(dim=0).tolist()}, std={normed2.std(dim=0).tolist()}")

    # Test LinearNormalizer
    ln = LinearNormalizer()
    ln.fit({'action': data, 'obs': torch.randn(100, 4)})
    assert 'action' in ln
    out = ln['action'].normalize(data)
    assert torch.allclose(out.min(dim=0).values, torch.tensor([-1.0, -1.0]))

    # Test state_dict round-trip
    sd = ln.state_dict()
    ln2 = LinearNormalizer()
    ln2.load_state_dict(sd)
    out2 = ln2['action'].normalize(data)
    assert torch.allclose(out, out2)

    print("LinearNormalizer OK")
