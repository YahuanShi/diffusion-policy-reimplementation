"""
Unit tests for the real-robot scheduling helpers (diffusion_policy/control.py).

Validates:
  - plan_execution never runs past the cycle boundary, whatever the latency
  - latency is converted into skipped slots
  - GripperHysteresis follows the dataset convention (1 = open), ignores values between
    the thresholds and respects the dwell time
"""

from diffusion_policy.control import GripperHysteresis, TimingStats, plan_execution

DT = 0.05  # 20 Hz


def test_plan_execution_no_latency():
    assert plan_execution(0.0, DT, 6, 8, 1000) == (0, 6)


def test_plan_execution_skips_late_slots_but_keeps_cycle_end():
    # 120 ms inference → slots 0, 1 already passed; still stop at slot 6
    assert plan_execution(0.12, DT, 6, 8, 1000) == (2, 6)


def test_plan_execution_overrun_executes_nothing():
    start, end = plan_execution(0.5, DT, 6, 8, 1000)
    assert start == end


def test_plan_execution_respects_max_steps():
    assert plan_execution(0.0, DT, 6, 8, 3) == (0, 3)


def test_plan_execution_clamps_to_predicted_chunk():
    assert plan_execution(0.0, DT, 10, 8, 1000) == (0, 8)


def test_gripper_hysteresis_band_does_not_toggle():
    g = GripperHysteresis(close_thresh=0.35, open_thresh=0.65, min_dwell_steps=1)
    for v in [0.5, 0.4, 0.6, 0.36]:
        assert g.update(v) is True  # stays open inside the band


def test_gripper_hysteresis_uses_dataset_convention():
    # Training data: 1 = open, 0 = closed. A policy output of ~1 must keep it open.
    g = GripperHysteresis(min_dwell_steps=1)
    assert all(g.update(1.0) for _ in range(10))
    assert g.update(0.0) is False


def test_gripper_hysteresis_switch_and_dwell():
    g = GripperHysteresis(close_thresh=0.35, open_thresh=0.65, min_dwell_steps=3)
    states = [g.update(v) for v in [0.1, 0.1, 0.1]]
    assert states == [True, True, False]  # closes once dwell is satisfied
    # Immediately asking to open is ignored until the dwell has passed again
    states = [g.update(1.0) for _ in range(3)]
    assert states == [False, False, True]


def test_gripper_hysteresis_reset():
    g = GripperHysteresis(min_dwell_steps=1)
    g.update(0.0)
    assert g.is_open is False
    g.reset()
    assert g.is_open is True


def test_timing_stats_summary():
    stats = TimingStats()
    for v in range(1, 101):  # 1..100 ms
        stats.add("obs", v)
    s = stats.summary()["obs"]
    assert s["n"] == 100 and s["max"] == 100
    assert s["mean"] == 50.5
    assert 94 <= s["p95"] <= 96
