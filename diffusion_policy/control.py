"""
Hardware-independent control helpers for real-robot deployment (used by inference.py).

Kept free of robot / camera / cv2 imports so the scheduling logic can be unit-tested in CI.

Timing model (one inference cycle, steps_per_inference=6, dt=1/20s):

  t_cycle_start                                          t_cycle_end
  |  obs  | inference ... |                                   |
  slot:  0       1        2       3       4       5           6 (= next cycle's slot 0)
                          ^ exec_start = int(elapsed / dt)

  action_chunk[i] is meant for slot i (action[0] = time of the obs).
  Inference latency eats the first slots, so execution starts at exec_start and
  ends at steps_per_inference — never past the cycle boundary. Running past it
  would make the next cycle start late, skip more, run further past, ... until
  only one action per cycle is executed.
"""

import time


def precise_wait(t_end, slack=0.001, time_func=time.monotonic):
    """Sleep until t_end with sub-millisecond accuracy (sleep + busy-wait)."""
    t_now = time_func()
    if t_now < t_end - slack:
        time.sleep(t_end - t_now - slack)
    while time_func() < t_end:
        pass


def plan_execution(elapsed, dt, steps_per_inference, n_action_steps, steps_remaining):
    """
    Decide which slice of the predicted action chunk to execute this cycle.

    elapsed:         seconds since the cycle's obs was captured (≈ inference latency)
    Returns:         (exec_start, exec_end) — execute actions[exec_start:exec_end].
                     exec_start == exec_end means inference overran the whole cycle.
    """
    exec_end = min(steps_per_inference, n_action_steps)
    exec_start = min(int(elapsed / dt), exec_end)
    exec_end = min(exec_end, exec_start + steps_remaining)
    return exec_start, exec_end


class GripperHysteresis:
    """
    Turn the continuous gripper channel of the action (≈0 open, ≈1 close) into a stable
    binary command.

    - Hysteresis: close only above close_thresh, open only below open_thresh, so values
      hovering around 0.5 don't toggle the gripper.
    - Dwell: after a switch, hold the new state for at least min_dwell_steps. Consecutive
      action chunks can disagree on the exact switch moment; the dwell absorbs that.
    """

    def __init__(self, close_thresh=0.65, open_thresh=0.35, min_dwell_steps=20):
        assert open_thresh < close_thresh
        self.close_thresh = close_thresh
        self.open_thresh = open_thresh
        self.min_dwell_steps = min_dwell_steps
        self.reset()

    def reset(self, is_open=True):
        self.is_open = is_open
        self._dwell = 0

    def update(self, raw):
        """Feed one raw gripper value; returns the effective state (True = open)."""
        self._dwell += 1
        if self._dwell >= self.min_dwell_steps:
            if self.is_open and raw >= self.close_thresh:
                self.is_open = False
                self._dwell = 0
            elif not self.is_open and raw <= self.open_thresh:
                self.is_open = True
                self._dwell = 0
        return self.is_open
