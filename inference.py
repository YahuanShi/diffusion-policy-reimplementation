"""
Real robot inference — deploy a trained diffusion policy on a UR3e robot.

Hardware:
    - UR3e via RTDE (configurable IP)
    - Weiss CRG 30-050 gripper via serial
    - Two Intel RealSense cameras (exterior + wrist)

Observation dict (matches LeRobot training format after safe_key renaming):
    observation_images_exterior_image_1_left  (B, T, 3, H, W) float32
    observation_images_wrist_image_left       (B, T, 3, H, W) float32
    observation_state                         (B, T, state_dim) float32

Action format: (7,) = [6 joint angles in DEGREES (model output, converted to rad before servoJ), 1 gripper (1=open, 0=closed — same as training data)]

Usage:
    python inference.py --checkpoint outputs/policy_final.pt --robot_ip 10.0.0.1
    python inference.py --checkpoint outputs/policy_final.pt --robot_ip 10.0.0.1 --dry_run
    python inference.py --checkpoint outputs/policy_final.pt --robot_ip 10.0.0.1 \
        --steps_per_inference 6 --num_inference_steps 16
    python inference.py --checkpoint outputs/policy_final.pt --robot_ip 10.0.0.1 \
        --record_dir trials/   # save videos + actions, then ask whether the trial succeeded
"""

import argparse
import logging
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
import torch

from diffusion_policy.control import (
    GripperHysteresis,
    TimingStats,
    plan_execution,
    precise_wait,
)
from diffusion_policy.policy.loader import load_policy

log = logging.getLogger(__name__)


# ══════════════════════════════ Configuration ══════════════════════════════

GRIPPER_PORT = "/dev/ttyACM0"
GRIPPER_BAUDRATE = 9600
GRIPPER_MAX_MM = 50
GRIPPER_OPEN_THRESH_MM = 5.0

# Gripper convention of the training data: 1 = open, 0 = closed (every demo starts and
# ends open). Both the state fed to the policy and the action channel use it.
GRIPPER_STATE_OPEN = 1.0
GRIPPER_STATE_CLOSED = 0.0

# Hysteresis thresholds for gripper state switching.
# action[6] must fall below CLOSE_THRESH to trigger close (when currently open),
# and rise above OPEN_THRESH to trigger open (when currently closed).
GRIPPER_CLOSE_THRESH = 0.35
GRIPPER_OPEN_THRESH = 0.65
# Minimum steps to hold gripper state before allowing another switch.
# Prevents chunk-boundary disagreement: consecutive inference cycles can produce
# conflicting gripper predictions; dwell time absorbs that noise.
# At 20 Hz: 20 steps = 1 second minimum hold.
GRIPPER_MIN_DWELL_STEPS = 20

CAM_SERIAL_EXTERIOR = "105422061000"
CAM_SERIAL_WRIST = "352122273671"
IMAGE_SIZE = 480
# A cached camera frame / gripper reading older than this means the reader stalled
MAX_SENSOR_AGE_S = 0.5

HOME_DEG = [45, -20, -140, -40, -270, 0]
HOME_RAD = np.radians(HOME_DEG).astype(np.float32)

SERVO_J_LOOKAHEAD = 0.2
SERVO_J_GAIN = 200
# Default safety velocity clamp (override with --max_joint_vel). With degree/radian
# units correct, normal demo speeds (≈ 30–60°/s ≈ 0.5–1.0 rad/s) pass through unclipped.
MAX_JOINT_VEL = 1.0  # rad/s

# Per-joint position limits (radians).  Tune these to match your actual workspace.
# Default: conservative limits that avoid common self-collision configurations on UR3e.
# Limits derived from actual training data ranges (action in degrees).
# Joint 4 operates in the -270° neighbourhood (3rd winding); generic ±175° would
# clip all predictions. Set per-joint limits to cover the data range plus margin.
#   j0: [15, 112]   j1: [-146, -19]   j2: [-141, -46]
#   j3: [-75, 2]    j4: [-289, -253]  j5: [-38, 38]
JOINT_LIMIT_LOW = np.radians([-180, -175, -170, -175, -310, -360]).astype(np.float32)
JOINT_LIMIT_HIGH = np.radians([180, 5, 170, 175, -220, 360]).astype(np.float32)


# ══════════════════════════════ Gripper ══════════════════════════════


def gripper_state_from_width(width_mm):
    """Measured jaw width → state value in the training-data convention (1 = open)."""
    return (
        GRIPPER_STATE_OPEN
        if width_mm > GRIPPER_OPEN_THRESH_MM
        else GRIPPER_STATE_CLOSED
    )


class WeissCRGGripper:
    """
    Weiss CRG 30-050 gripper driver via DC-IOLink USB adapter.

    Protocol: PDOUT=[02,00] open | PDOUT=[03,00] close | PDOUT=[07,00] reference
    Feedback: @PDIN=[B0,B1,B2,B3] where pos_mm = ((B0<<8)|B1)/100

    A background thread drains the serial port continuously and caches the latest
    @PDIN reading. Reading inside the control loop instead would block for up to the
    serial timeout and, worse, return the *oldest* buffered line — a stale state.
    """

    FLAG_OPEN = 1
    FLAG_CLOSED = 2
    READ_TIMEOUT_S = 0.02  # max time the reader holds the lock per readline

    def __init__(self, port=GRIPPER_PORT, baudrate=GRIPPER_BAUDRATE):
        import serial as _serial

        self._lock = threading.Lock()
        self._ser = _serial.Serial(port=port, baudrate=baudrate, timeout=0.2)
        self._position_mm = 0.0
        self._flags = 0
        self._last_update = 0.0
        log.info(f"[Gripper] Opened {port} @ {baudrate}")
        self._initialise()

        self._ser.timeout = self.READ_TIMEOUT_S
        self._stop_event = threading.Event()
        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()

    def _send(self, cmd, wait=0.3):
        with self._lock:
            try:
                self._ser.reset_input_buffer()
                self._ser.write((cmd + "\n").encode("ascii"))
            except Exception as e:
                log.warning(f"[Gripper] Serial error '{cmd}': {e}")
        time.sleep(wait)

    def _parse_pdin(self, line):
        try:
            inner = line[7:].split("]")[0]
            parts = [int(x, 16) for x in inner.split(",")]
            self._position_mm = ((parts[0] << 8) | parts[1]) / 100.0
            self._flags = parts[3] if len(parts) >= 4 else 0
            self._last_update = time.monotonic()
            return True
        except (ValueError, IndexError):
            return False

    def _reader_loop(self):
        while not self._stop_event.is_set():
            with self._lock:
                try:
                    raw = self._ser.readline() if self._ser.is_open else b""
                except Exception as e:
                    log.warning(f"[Gripper] Read error: {e}")
                    raw = b""
            line = raw.decode("ascii", errors="ignore").strip()
            if line.startswith("@PDIN=["):
                self._parse_pdin(line)
            elif not raw:
                time.sleep(0.001)  # let _send grab the lock between empty reads

    def _wait_flag(self, flag_bit, timeout=6.0):
        """Wait for a flag in a reading that arrived *after* this call started."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self._last_update > t0 and (self._flags & (1 << flag_bit)):
                return True
            time.sleep(0.02)
        return False

    def _set_positions(self, open_mm=GRIPPER_MAX_MM, close_mm=0.5):
        def enc(mm):
            v = int(mm * 100)
            return f"[{(v >> 8) & 0xFF:02x},{v & 0xFF:02x}]"

        self._send(f"SETPARAM(96, 2, {enc(open_mm)})", 0.3)
        self._send(f"SETPARAM(96, 1, {enc(close_mm)})", 0.3)
        self._send("SETPARAM(96, 3, [64])", 0.3)

    def _initialise(self):
        for cmd in ["ID?", "ID?", "FALLBACK(1)", "MODE?", "RESTART()", "OPERATE()"]:
            self._send(cmd, 0.5)
        self._send("PDOUT=[00,00]", 0.5)
        # Configure open/close positions once — no need to repeat before each move
        self._set_positions(GRIPPER_MAX_MM, 0.5)
        log.info("[Gripper] Initialised.")

    def home(self):
        log.info("[Gripper] Homing...")
        self._set_positions(GRIPPER_MAX_MM, 0.5)
        self._send("PDOUT=[07,00]", 0.2)
        if not self._wait_flag(self.FLAG_OPEN, timeout=10.0):
            log.warning("[Gripper] No 'open' flag after homing — check the gripper")
        log.info("[Gripper] Home complete — open.")

    def move_to_pos(self, width_mm):
        # _set_positions already called once at init; just send the move command.
        # No blocking wait — gripper moves in the background while the control loop continues.
        if width_mm > GRIPPER_OPEN_THRESH_MM:
            self._send("PDOUT=[02,00]", 0.0)
        else:
            self._send("PDOUT=[03,00]", 0.0)

    def get_width(self):
        """Latest cached width in mm and its age in seconds (never blocks)."""
        return self._position_mm, time.monotonic() - self._last_update

    def close(self):
        self._stop_event.set()
        self._reader.join(timeout=1.0)
        # Zero all outputs to cut drive, then close serial immediately.
        # Do NOT send FALLBACK(1) here — it activates the gripper's internal
        # fallback program which causes it to cycle open/close autonomously.
        self._send("PDOUT=[00,00]", 0.1)
        with self._lock:
            if self._ser and self._ser.is_open:
                self._ser.close()
        log.info("[Gripper] Closed.")


# ══════════════════════════════ Camera ══════════════════════════════


def _center_crop_resize(bgr, size=IMAGE_SIZE):
    h, w = bgr.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    crop = bgr[y0 : y0 + s, x0 : x0 + s]
    resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


class RealSenseCamera:
    """
    One RealSense color stream captured by a background thread.

    wait_for_frames() blocks until the next frame (up to 33 ms at 30 fps); doing that
    for two cameras inside get_obs() would eat most of a 50 ms control period. The
    thread keeps only the latest frame, so read() returns immediately.
    """

    def __init__(self, serial_num, width=640, height=480, fps=30):
        import pyrealsense2 as rs

        self.serial = serial_num
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial_num)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self._pipeline.start(cfg)
        self._frame = None
        self._t_frame = 0.0
        self._frame_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        log.info(f"[Camera] Started — serial {serial_num}")

    def _capture_loop(self):
        while not self._stop_event.is_set():
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=500)
            except RuntimeError as e:
                log.warning(f"[Camera {self.serial}] {e}")
                continue
            color = frames.get_color_frame()
            if color:
                bgr = np.asanyarray(color.get_data()).copy()
                with self._frame_lock:
                    self._frame, self._t_frame = bgr, time.monotonic()

    def read(self, timeout=2.0):
        """Latest BGR frame and its age in seconds. Waits only for the very first frame."""
        t0 = time.monotonic()
        while True:
            with self._frame_lock:
                frame, t_frame = self._frame, self._t_frame
            if frame is not None:
                return frame, time.monotonic() - t_frame
            if time.monotonic() - t0 > timeout:
                raise RuntimeError(f"Camera {self.serial} produced no frame")
            time.sleep(0.005)

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        self._pipeline.stop()


# ══════════════════════════════ Robot ══════════════════════════════


class UR3eRobot:
    """UR3e robot + Weiss gripper + dual RealSense cameras."""

    def __init__(
        self,
        robot_ip,
        frequency=20,
        use_gripper=True,
        cam_serial_exterior=CAM_SERIAL_EXTERIOR,
        cam_serial_wrist=CAM_SERIAL_WRIST,
        image_size=IMAGE_SIZE,
        max_joint_vel=MAX_JOINT_VEL,
    ):
        import rtde_control
        import rtde_receive

        self._dt = 1.0 / frequency
        self._image_size = image_size
        self._max_joint_vel = max_joint_vel

        log.info(f"[UR3e] Connecting RTDE to {robot_ip}...")
        self._rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        self._rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        log.info("[UR3e] RTDE connected.")

        self._gripper = None
        self._last_gripper_open = None
        if use_gripper:
            log.info(f"[UR3e] Connecting gripper on {GRIPPER_PORT}...")
            self._gripper = WeissCRGGripper()
            self._last_gripper_open = True

        log.info("[UR3e] Starting cameras...")
        self._cam_exterior = RealSenseCamera(cam_serial_exterior)
        self._cam_wrist = RealSenseCamera(cam_serial_wrist)

        self._last_cmd_rad = HOME_RAD.copy()

    def home(self):
        import contextlib

        with contextlib.suppress(Exception):
            self._rtde_c.servoStop()
        time.sleep(0.2)
        log.info("[UR3e] Moving to home...")
        self._rtde_c.moveJ(HOME_RAD.tolist(), speed=0.25, acceleration=0.25)
        self._last_cmd_rad = HOME_RAD.copy()
        if self._gripper:
            self._gripper.home()
            self._last_gripper_open = True
        log.info("[UR3e] Ready.")

    def get_obs(self):
        """
        Assemble one observation from cached sensor readings (non-blocking).

        Returns the obs dict plus "sensor_age_s": the age of the oldest reading used.
        Raises if a sensor stalled, so the robot stops instead of acting on stale data.
        """
        qpos = np.array(self._rtde_r.getActualQ(), dtype=np.float32)

        # No gripper attached → report "open", the state every demonstration starts in
        gripper_state = np.array([GRIPPER_STATE_OPEN], dtype=np.float32)
        ages = []
        if self._gripper:
            width, gripper_age = self._gripper.get_width()
            ages.append(gripper_age)
            gripper_state[0] = gripper_state_from_width(width)

        # Training data stores joint angles in DEGREES — normalizer was fitted on degrees.
        # getActualQ() returns radians, so convert before building the state vector.
        state = np.concatenate([np.degrees(qpos), gripper_state])

        exterior_bgr, ext_age = self._cam_exterior.read()
        wrist_bgr, wrist_age = self._cam_wrist.read()
        ages += [ext_age, wrist_age]
        sensor_age = max(ages)
        if sensor_age > MAX_SENSOR_AGE_S:
            raise RuntimeError(
                f"Stale sensor data ({sensor_age * 1000:.0f} ms old) — "
                "camera or gripper reader stalled"
            )

        return {
            "state": state,
            "exterior_rgb": _center_crop_resize(exterior_bgr, self._image_size),
            "wrist_rgb": _center_crop_resize(wrist_bgr, self._image_size),
            "sensor_age_s": sensor_age,
        }

    def apply_action(self, action, gripper_open):
        """
        action:       (7,) array — [6 joint angles DEGREES (model output), 1 gripper raw]
        gripper_open: effective gripper command after hysteresis (see GripperHysteresis)
        Converts degrees → radians before servoJ. Velocity-limited + joint-clamped.
        """
        # Model outputs degrees (training data stored in degrees); convert before sending to RTDE
        target_rad = np.radians(action[:6])
        # Clamp to joint limits before velocity-limiting to prevent self-collision
        target_rad = np.clip(target_rad, JOINT_LIMIT_LOW, JOINT_LIMIT_HIGH)
        max_step = self._max_joint_vel * self._dt
        delta = target_rad - self._last_cmd_rad
        cmd_rad = self._last_cmd_rad + np.clip(delta, -max_step, max_step)
        self._last_cmd_rad = cmd_rad.copy()

        # SERVO_J_TIME must equal the control loop period so RTDE trajectory
        # interpolation matches the actual step rate.
        self._rtde_c.servoJ(
            cmd_rad.tolist(), 0, 0, self._dt, SERVO_J_LOOKAHEAD, SERVO_J_GAIN
        )

        if self._gripper and gripper_open != self._last_gripper_open:
            width_mm = GRIPPER_MAX_MM if gripper_open else 0.0
            self._gripper.move_to_pos(width_mm)
            self._last_gripper_open = gripper_open
            log.info(
                f"[UR3e] Gripper {'opening' if gripper_open else 'closing'} (raw={action[6]:.2f})"
            )

    def stop(self):
        try:
            self._rtde_c.servoStop()
            self._rtde_c.stopScript()
        except Exception:
            pass
        if self._gripper:
            self._gripper.close()
        for cam in (self._cam_exterior, self._cam_wrist):
            try:
                cam.stop()
            except Exception as e:
                log.warning(f"[Camera {cam.serial}] stop failed: {e}")
        log.info("[UR3e] Stopped.")


def make_vis_frame(obs, action, step, inference_ms, eff_gripper_open=True):
    """Side-by-side exterior + wrist camera view with status overlay."""
    ext = cv2.cvtColor(obs["exterior_rgb"], cv2.COLOR_RGB2BGR)
    wrist = cv2.cvtColor(obs["wrist_rgb"], cv2.COLOR_RGB2BGR)
    disp = 320
    frame = np.concatenate(
        [cv2.resize(ext, (disp, disp)), cv2.resize(wrist, (disp, disp))], axis=1
    )
    gripper_cmd = "open" if eff_gripper_open else "close"
    gripper_actual = "open" if obs["state"][-1] > 0.5 else "closed"
    lines = [
        f"Step: {step}",
        f"Infer: {inference_ms:.0f}ms",
        f"G raw={action[6]:.2f} cmd={gripper_cmd} actual={gripper_actual}",
        "q / ESC — stop",
    ]
    for i, txt in enumerate(lines):
        cv2.putText(
            frame,
            txt,
            (10, 22 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return frame


# ══════════════════════════════ Inference Loop ══════════════════════════════


def make_obs_key_map(shape_meta):
    """
    Map policy obs keys (from the checkpoint's shape_meta) to UR3eRobot.get_obs() keys.

    e.g. observation_images_exterior_image_1_left → exterior_rgb
         observation_images_wrist_image_left      → wrist_rgb
         observation_state                        → state
    Fails loudly if the checkpoint expects an observation this robot can't provide.
    """
    key_map = {}
    for key, attr in shape_meta["obs"].items():
        if attr.get("type") == "low_dim":
            key_map[key] = "state"
        elif "exterior" in key:
            key_map[key] = "exterior_rgb"
        elif "wrist" in key:
            key_map[key] = "wrist_rgb"
        else:
            raise ValueError(f"No robot observation matches policy key '{key}'")
    return key_map


def build_obs_dict(obs_history, key_map, device="cuda"):
    """
    Convert raw observation history to policy input tensors.

    obs_history: deque of dicts from UR3eRobot.get_obs(), length = n_obs_steps
    key_map:     policy key → robot obs key (from make_obs_key_map)
    Returns: obs_dict with (1, n_obs_steps, ...) tensors, keys matching training format.
    """
    obs_dict = {}
    for policy_key, robot_key in key_map.items():
        stack = np.stack([o[robot_key] for o in obs_history])  # (T, ...)
        t = torch.from_numpy(stack).float()
        if robot_key != "state":
            # HWC uint8 -> CHW float32 [0, 1]
            t = t.permute(0, 3, 1, 2) / 255.0
        obs_dict[policy_key] = t.unsqueeze(0).to(device)
    return obs_dict


def run_inference(
    policy,
    robot,
    shape_meta,
    device="cuda",
    steps_per_inference=6,
    max_steps=800,
    frequency=20,
    dry_run=False,
    recorder=None,
):
    """
    Run the receding-horizon control loop until max_steps or the user stops it.

    recorder: optional evaluation.trial_recorder.TrialRecorder — every executed step is
    appended to it. Returns the TimingStats of the run.
    """
    dt = 1.0 / frequency
    frame_latency = 1.0 / 30  # camera runs at 30 fps
    n_obs_steps = policy.n_obs_steps
    n_action_steps = policy.n_action_steps
    key_map = make_obs_key_map(shape_meta)
    obs_history = deque(maxlen=n_obs_steps)

    # Fill obs history before first inference
    log.info("Warming up cameras...")
    for _ in range(n_obs_steps):
        obs_history.append(robot.get_obs())
        time.sleep(dt)

    # Fail before moving if the robot's obs don't match what the policy was trained on
    for policy_key, robot_key in key_map.items():
        expected = tuple(shape_meta["obs"][policy_key]["shape"])
        got = obs_history[-1][robot_key].shape
        if robot_key != "state":
            got = (got[2], got[0], got[1])  # HWC → CHW
        if got != expected:
            raise ValueError(
                f"Obs '{robot_key}' has shape {got}, policy '{policy_key}' expects {expected}"
            )

    # One warm-up inference pass to trigger CUDA kernel compilation
    log.info("Warming up policy inference...")
    with torch.no_grad():
        policy.predict_action(build_obs_dict(obs_history, key_map, device))
    log.info(
        f"Ready. freq={frequency}Hz  steps_per_inference={steps_per_inference}  "
        f"max_steps={max_steps}  dry_run={dry_run}"
    )

    step = 0
    iter_idx = 0
    # Single source of truth for the gripper command (robot + visualisation)
    gripper = GripperHysteresis(
        close_thresh=GRIPPER_CLOSE_THRESH,
        open_thresh=GRIPPER_OPEN_THRESH,
        min_dwell_steps=GRIPPER_MIN_DWELL_STEPS,
    )
    # obs: get_obs() duration | late: how far past its slot an action was sent
    # age: age of the oldest sensor reading | infer: policy.predict_action()
    timing = TimingStats()
    t_start = time.monotonic()

    try:
        while step < max_steps:
            # Absolute start time of this inference cycle on the global clock
            t_cycle_start = t_start + iter_idx * dt
            t_cycle_end = t_cycle_start + steps_per_inference * dt

            # ── Capture obs aligned to cycle start ──────────────────────────
            # Take a fresh obs snapshot at the scheduled cycle boundary so the
            # model always sees obs from a consistent, predictable point in time.
            precise_wait(t_cycle_start)
            t_obs = time.monotonic()
            obs = robot.get_obs()
            timing.add("obs", (time.monotonic() - t_obs) * 1000)
            obs_history.append(obs)

            obs_dict = build_obs_dict(obs_history, key_map, device)

            # ── Inference ───────────────────────────────────────────────────
            t_inf_start = time.monotonic()
            with torch.no_grad():
                action_chunk = policy.predict_action(obs_dict)
            inference_ms = (time.monotonic() - t_inf_start) * 1000
            timing.add("infer", inference_ms)

            actions = action_chunk[0].cpu().numpy()  # (n_action_steps, action_dim)

            # ── Latency compensation (key improvement from original project) ─
            # Count how many dt-slots inference consumed and skip those actions, so
            # execution starts from the slot that matches the current time. Execution
            # always ends at the cycle boundary (see diffusion_policy/control.py).
            elapsed_since_cycle = time.monotonic() - t_cycle_start
            exec_start, exec_end = plan_execution(
                elapsed_since_cycle,
                dt,
                steps_per_inference,
                n_action_steps,
                max_steps - step,
            )
            cycle_obs_ms, cycle_late_ms, cycle_age_ms = [], [], []
            if exec_start == exec_end:
                log.warning(
                    f"Inference ({inference_ms:.0f}ms) overran the whole cycle "
                    f"({steps_per_inference * dt * 1000:.0f}ms) — no actions executed"
                )

            # ── Execute action chunk ─────────────────────────────────────────
            for i in range(exec_start, exec_end):
                # Each action fires at its absolute scheduled time slot
                t_slot = t_cycle_start + i * dt
                precise_wait(t_slot)
                cycle_late_ms.append((time.monotonic() - t_slot) * 1000)

                gripper_open = gripper.update(float(actions[i][6]))
                if not dry_run:
                    robot.apply_action(actions[i], gripper_open)

                t_obs = time.monotonic()
                obs = robot.get_obs()
                obs_ms = (time.monotonic() - t_obs) * 1000
                cycle_obs_ms.append(obs_ms)
                cycle_age_ms.append(obs["sensor_age_s"] * 1000)
                obs_history.append(obs)
                if recorder is not None:
                    recorder.add_step(
                        obs, actions[i], gripper_open, inference_ms, obs_ms
                    )

                # Visualize — non-blocking
                cv2.imshow(
                    "Diffusion Policy",
                    make_vis_frame(obs, actions[i], step, inference_ms, gripper_open),
                )
                key = cv2.pollKey()
                if key in (ord("q"), 27):  # q or ESC
                    raise KeyboardInterrupt

                step += 1

            for name, vals in (
                ("obs", cycle_obs_ms),
                ("late", cycle_late_ms),
                ("age", cycle_age_ms),
            ):
                for v in vals:
                    timing.add(name, v)
            max_obs = max(cycle_obs_ms, default=0.0)
            log.info(
                f"step {step:>4}: infer={inference_ms:.0f}ms skip={exec_start} "
                f"exec={exec_end - exec_start} obs_max={max_obs:.0f}ms "
                f"late_max={max(cycle_late_ms, default=0.0):.0f}ms "
                f"age_max={max(cycle_age_ms, default=0.0):.0f}ms"
            )
            if max_obs > 0.5 * dt * 1000:
                log.warning(
                    f"get_obs took {max_obs:.0f}ms (> half the {dt * 1000:.0f}ms period)"
                )

            now = time.monotonic()
            if now > t_cycle_end:
                # Overran the cycle (very slow inference): re-anchor the schedule on the
                # next free slot instead of carrying the delay into every later cycle.
                iter_idx = int(np.ceil((now - t_start) / dt))
            else:
                # Wait until this cycle's deadline before next inference
                precise_wait(t_cycle_end - frame_latency)
                iter_idx += steps_per_inference

    except KeyboardInterrupt:
        log.info("Stopped by user.")
    except Exception as e:
        log.error(f"[Fatal] Pipeline error at step {step}: {e}")
        log.error("Stopping robot and exiting.")
        # sys.exit raises SystemExit (BaseException), so finally still runs
        # to call robot.stop(), then the process exits.
        sys.exit(1)
    finally:
        cv2.destroyAllWindows()
        elapsed = time.monotonic() - t_start
        rate = step / elapsed if elapsed > 0 else 0.0
        log.info(
            f"Inference ended at step {step}. Effective rate {rate:.1f}Hz "
            f"(target {frequency}Hz)"
        )
        log.info(f"Timing: {timing.format()}")
    return timing


# ══════════════════════════════ Main ══════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="UR3e real robot inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--robot_ip", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--frequency",
        type=int,
        default=None,
        help="Control rate in Hz (default: the training dataset's fps). Must match the "
        "dataset fps — obs spacing and per-step actions were learned at that rate.",
    )
    parser.add_argument(
        "--allow_fps_mismatch",
        action="store_true",
        help="Run even if --frequency differs from the dataset fps",
    )
    parser.add_argument(
        "--steps_per_inference",
        type=int,
        default=6,
        help="Actions to execute per inference cycle (default: 6)",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=16,
        help="DDIM denoising steps at inference (default: 16)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=800,
        help="Control steps per trial (default: 800 = 40s at 20Hz; longest demo is ~33s)",
    )
    parser.add_argument(
        "--record_dir",
        default=None,
        help="Save each trial (videos, actions, timing) here and ask for the outcome",
    )
    parser.add_argument(
        "--max_joint_vel",
        type=float,
        default=MAX_JOINT_VEL,
        help=f"Per-joint velocity limit in rad/s (default: {MAX_JOINT_VEL})",
    )
    parser.add_argument("--no_gripper", action="store_true")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run full pipeline without sending robot commands",
    )
    parser.add_argument(
        "--cam_exterior",
        default=CAM_SERIAL_EXTERIOR,
        help="RealSense serial number for exterior camera",
    )
    parser.add_argument(
        "--cam_wrist",
        default=CAM_SERIAL_WRIST,
        help="RealSense serial number for wrist camera",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)

    log.info("Loading policy...")
    policy, shape_meta, config = load_policy(
        args.checkpoint, args.device, num_inference_steps=args.num_inference_steps
    )
    dataset_fps = config.get("fps")
    log.info(
        f"Policy loaded. Action dim: {shape_meta['action']['shape'][0]}, "
        f"DDIM steps: {args.num_inference_steps}, dataset fps: {dataset_fps}"
    )

    frequency = args.frequency
    if frequency is None:
        if dataset_fps is None:
            parser.error("checkpoint does not record the dataset fps; pass --frequency")
        frequency = dataset_fps
    elif dataset_fps is not None and frequency != dataset_fps:
        msg = (
            f"--frequency {frequency} differs from the dataset fps {dataset_fps}: "
            "the robot would move at the wrong speed and obs frames would be "
            "spaced differently than in training."
        )
        if not args.allow_fps_mismatch:
            parser.error(msg + " Pass --allow_fps_mismatch to run anyway.")
        log.warning(msg)

    log.info(f"Connecting to UR3e at {args.robot_ip}...")
    robot = UR3eRobot(
        robot_ip=args.robot_ip,
        frequency=frequency,
        use_gripper=not args.no_gripper,
        cam_serial_exterior=args.cam_exterior,
        cam_serial_wrist=args.cam_wrist,
        max_joint_vel=args.max_joint_vel,
    )

    recorder = None
    if args.record_dir:
        from evaluation.trial_recorder import TrialRecorder

        recorder = TrialRecorder(
            args.record_dir,
            fps=frequency,
            meta={
                "checkpoint": args.checkpoint,
                "frequency": frequency,
                "steps_per_inference": args.steps_per_inference,
                "num_inference_steps": args.num_inference_steps,
                "max_steps": args.max_steps,
                "dry_run": args.dry_run,
            },
        )
        log.info(f"Recording trial to {recorder.trial_dir}")

    try:
        # A dry run must never move the robot — not even to home
        if args.dry_run:
            log.info("[dry run] Skipping homing — robot and gripper will not move.")
        else:
            robot.home()

        run_inference(
            policy=policy,
            robot=robot,
            shape_meta=shape_meta,
            device=args.device,
            frequency=frequency,
            steps_per_inference=args.steps_per_inference,
            max_steps=args.max_steps,
            dry_run=args.dry_run,
            recorder=recorder,
        )
    finally:
        # Always release cameras and serial port (also in a dry run)
        robot.stop()
        if recorder is not None:
            from evaluation.trial_recorder import ask_trial_outcome

            success, notes = ask_trial_outcome()
            log.info(f"Saved trial to {recorder.close(success, notes)}")


if __name__ == "__main__":
    main()
