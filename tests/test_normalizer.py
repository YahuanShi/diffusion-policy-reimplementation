"""
Unit tests for LinearNormalizer.

Validates:
  - fit + normalize + unnormalize is a round-trip (lossless)
  - Output range after normalize is within [-1, 1]
  - Multiple keys are handled independently
  - Single-element tensors don't cause division-by-zero
"""

import torch
from diffusion_policy.dataset.normalizer import LinearNormalizer


def test_round_trip():
    normalizer = LinearNormalizer()
    data = torch.randn(100, 4) * 5 + 3
    normalizer.fit({"action": data})

    normalized = normalizer["action"].normalize(data)
    recovered = normalizer["action"].unnormalize(normalized)
    assert torch.allclose(data, recovered, atol=1e-5), "normalize → unnormalize must be lossless"


def test_output_range():
    normalizer = LinearNormalizer()
    data = torch.randn(200, 6) * 10
    normalizer.fit({"action": data})

    normalized = normalizer["action"].normalize(data)
    assert normalized.min().item() >= -1.0 - 1e-5
    assert normalized.max().item() <= 1.0 + 1e-5


def test_multiple_keys_independent():
    normalizer = LinearNormalizer()
    action = torch.randn(100, 2) * 3
    state = torch.randn(100, 4) * 0.1
    normalizer.fit({"action": action, "state": state})

    norm_action = normalizer["action"].normalize(action)
    norm_state = normalizer["state"].normalize(state)

    # Both should map to [-1, 1]
    assert norm_action.abs().max().item() <= 1.0 + 1e-5
    assert norm_state.abs().max().item() <= 1.0 + 1e-5

    # Keys must not interfere
    recovered_action = normalizer["action"].unnormalize(norm_action)
    assert torch.allclose(action, recovered_action, atol=1e-5)


def test_key_membership():
    normalizer = LinearNormalizer()
    normalizer.fit({"action": torch.randn(50, 2)})
    assert "action" in normalizer
    assert "state" not in normalizer


def test_constant_input_no_crash():
    normalizer = LinearNormalizer()
    # All-same values: range=0 — normalizer should handle this without NaN
    data = torch.ones(50, 2)
    normalizer.fit({"action": data})
    normalized = normalizer["action"].normalize(data)
    assert not torch.isnan(normalized).any(), "constant input must not produce NaN"
