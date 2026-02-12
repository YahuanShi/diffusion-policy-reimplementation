"""
Unit tests for the warmup + cosine LR multiplier (training/workspace.py).
"""

import pytest

from training.workspace import make_lr_lambda


def test_warmup_is_linear():
    f = make_lr_lambda(warmup_steps=4, total_steps=100)
    assert [f(s) for s in range(4)] == pytest.approx([0.25, 0.5, 0.75, 1.0])


def test_cosine_decays_to_min_ratio_and_stays_flat():
    f = make_lr_lambda(warmup_steps=10, total_steps=110, min_ratio=0.1)
    assert f(10) == pytest.approx(1.0)
    assert f(60) == pytest.approx(0.55)  # halfway through the decay
    assert f(110) == pytest.approx(0.1)
    assert f(500) == pytest.approx(0.1)
