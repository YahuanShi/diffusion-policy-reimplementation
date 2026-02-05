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

Action format: (7,) = [6 joint angles in rad (absolute), 1 gripper (0=open, 1=close)]

Usage:
    python inference.py --checkpoint outputs/policy_final.pt --robot_ip 10.0.0.1
    python inference.py --checkpoint outputs/policy_final.pt --robot_ip 10.0.0.1 --dry_run
    python inference.py --checkpoint outputs/policy_final.pt --robot_ip 10.0.0.1 \
        --frequency 10 --steps_per_inference 6 --num_inference_steps 16
"""

import argparse
import logging
import threading
import time
from collections import deque

import cv2
import numpy as np
import torch

from diffusion_policy.model.diffusion.scheduler import DDIMScheduler
from diffusion_policy.model.vision.encoder import MultiImageObsEncoder
from diffusion_policy.dataset.normalizer import LinearNormalizer
from diffusion_policy.policy.image import DiffusionUnetImagePolicy

log = logging.getLogger(__name__)


def precise_wait(t_end, slack=0.001, time_func=time.monotonic):
    """Sleep until t_end with sub-millisecond accuracy (sleep + busy-wait)."""
    t_now = time_func()
    if t_now < t_end - slack:
        time.sleep(t_end - t_now - slack)
    while time_func() < t_end:
        pass


# ══════════════════════════════ Configuration ══════════════════════════════

GRIPPER_PORT = "/dev/ttyACM0"
GRIPPER_BAUDRATE = 9600
GRIPPER_MAX_MM = 50
GRIPPER_OPEN_THRESH_MM = 5.0

CAM_SERIAL_EXTERIOR = "105422061000"
CAM_SERIAL_WRIST = "352122273671"
IMAGE_SIZE = 480

HOME_DEG = [0.0, -90.0, 0.0, -90.0, 0.0, 0.0]
HOME_RAD = np.radians(HOME_DEG).astype(np.float32)

SERVO_J_TIME = 0.1
SERVO_J_LOOKAHEAD = 0.2
SERVO_J_GAIN = 200
MAX_JOINT_VEL = 0.8  # rad/s safety clamp


# ══════════════════════════════ Gripper ══════════════════════════════

class WeissCRGGripper:
    """
    Weiss CRG 30-050 gripper driver via DC-IOLink USB adapter.

    Protocol: PDOUT=[02,00] open | PDOUT=[03,00] close | PDOUT=[07,00] reference
    Feedback: @PDIN=[B0,B1,B2,B3] where pos_mm = ((B0<<8)|B1)/100
    """

    FLAG_OPEN = 1
    FLAG_CLOSED = 2

    def __init__(self, port=GRIPPER_PORT, baudrate=GRIPPER_BAUDRATE):
        import serial as _serial
        self._lock = threading.Lock()
        self._ser = _serial.Serial(port=port, baudrate=baudrate, timeout=0.2)
        self._position_mm = 0.0
        self._flags = 0
        log.info(f"[Gripper] Opened {port} @ {baudrate}")
        self._initialise()

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
            return True
        except Exception:
            return False

    def _read_pdin(self, timeout=1.0):
        t0 = time.monotonic()
        with self._lock:
            saved, self._ser.timeout = self._ser.timeout, 0.15
        try:
            while time.monotonic() - t0 < timeout:
                with self._lock:
                    line = self._ser.readline().decode("ascii", errors="ignore").strip()
                if line.startswith("@PDIN=[") and self._parse_pdin(line):
                    return True
        finally:
            with self._lock:
                self._ser.timeout = saved
        return False

    def _wait_flag(self, flag_bit, timeout=6.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self._read_pdin(0.5) and (self._flags & (1 << flag_bit)):
                return True
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
        log.info("[Gripper] Initialised.")

    def home(self):
        log.info("[Gripper] Homing...")
        self._set_positions(GRIPPER_MAX_MM, 0.5)
        self._send("PDOUT=[07,00]", 0.2)
        self._wait_flag(self.FLAG_OPEN, timeout=10.0)
        log.info("[Gripper] Home complete — open.")

    def move_to_pos(self, width_mm):
        self._set_positions(GRIPPER_MAX_MM, 0.5)
        if width_mm > GRIPPER_OPEN_THRESH_MM:
            self._send("PDOUT=[02,00]", 0.2)
        else:
            self._send("PDOUT=[03,00]", 0.2)

    def get_width(self):
        self._read_pdin(timeout=0.08)
        return self._position_mm

    def close(self):
        self._send("PDOUT=[00,00]", 0.3)
        self._send("FALLBACK(1)", 0.3)
        with self._lock:
            if self._ser and self._ser.is_open:
                self._ser.close()
        log.info("[Gripper] Closed.")


# ══════════════════════════════ Camera ══════════════════════════════

def _center_crop_resize(bgr, size=IMAGE_SIZE):
    h, w = bgr.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    crop = bgr[y0:y0+s, x0:x0+s]
    resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def _start_realsense(serial_num):
    import pyrealsense2 as rs
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial_num)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(cfg)
    log.info(f"[Camera] Started — serial {serial_num}")
    return pipeline


def _grab_frame(pipeline, retries=3):
    for attempt in range(retries):
        frames = pipeline.wait_for_frames(timeout_ms=500)
        color = frames.get_color_frame()
        if color:
            return np.asanyarray(color.get_data())
        log.warning(f"[Camera] No frame, retry {attempt+1}/{retries}")
    raise RuntimeError("Camera failed after retries")


# ══════════════════════════════ Robot ══════════════════════════════

class UR3eRobot:
    """UR3e robot + Weiss gripper + dual RealSense cameras."""

    def __init__(self, robot_ip, frequency=10, use_gripper=True,
                 cam_serial_exterior=CAM_SERIAL_EXTERIOR,
                 cam_serial_wrist=CAM_SERIAL_WRIST,
                 image_size=IMAGE_SIZE):
        import rtde_control
        import rtde_receive

        self._dt = 1.0 / frequency
        self._image_size = image_size

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
        self._pipe_exterior = _start_realsense(cam_serial_exterior)
        self._pipe_wrist = _start_realsense(cam_serial_wrist)

        self._last_cmd_rad = HOME_RAD.copy()

    def home(self):
        import contextlib
        with contextlib.suppress(Exception):
            self._rtde_c.servoStop()
        time.sleep(0.2)
        log.info("[UR3e] Moving to home...")
        self._rtde_c.moveJ(HOME_RAD.tolist(), speed=0.5, acceleration=0.5)
        self._last_cmd_rad = HOME_RAD.copy()
        if self._gripper:
            self._gripper.home()
            self._last_gripper_open = True
        log.info("[UR3e] Ready.")

    def get_obs(self):
        qpos = np.array(self._rtde_r.getActualQ(), dtype=np.float32)

        gripper_state = np.array([0.0], dtype=np.float32)
        if self._gripper:
            width = self._gripper.get_width()
            gripper_state[0] = 0.0 if width > GRIPPER_OPEN_THRESH_MM else 1.0

        state = np.concatenate([qpos, gripper_state])

        exterior_bgr = _grab_frame(self._pipe_exterior)
        wrist_bgr = _grab_frame(self._pipe_wrist)
        exterior_rgb = _center_crop_resize(exterior_bgr, self._image_size)
        wrist_rgb = _center_crop_resize(wrist_bgr, self._image_size)

        return {
            'state': state,
            'exterior_rgb': exterior_rgb,
            'wrist_rgb': wrist_rgb,
        }

    def apply_action(self, action):
        """
        action: (7,) array — [6 joint angles rad, 1 gripper 0/1]
        Velocity-limited servoJ + binary gripper control.
        """
        target_rad = action[:6]
        max_step = MAX_JOINT_VEL * self._dt
        delta = target_rad - self._last_cmd_rad
        cmd_rad = self._last_cmd_rad + np.clip(delta, -max_step, max_step)
        self._last_cmd_rad = cmd_rad.copy()

        try:
            self._rtde_c.servoJ(
                cmd_rad.tolist(), 0, 0,
                SERVO_J_TIME, SERVO_J_LOOKAHEAD, SERVO_J_GAIN)
        except Exception as e:
            log.warning(f"[UR3e] servoJ error: {e}")
            self._last_cmd_rad = np.array(
                self._rtde_r.getActualQ(), dtype=np.float32)

        if self._gripper:
            want_open = float(action[6]) < 0.5
            if want_open != self._last_gripper_open:
                width_mm = GRIPPER_MAX_MM if want_open else 0.0
                self._gripper.move_to_pos(width_mm)
                self._last_gripper_open = want_open
                log.info(f"[UR3e] Gripper {'opening' if want_open else 'closing'}")

    def stop(self):
        try:
            self._rtde_c.servoStop()
            self._rtde_c.stopScript()
        except Exception:
            pass
        if self._gripper:
            self._gripper.close()
        try:
            self._pipe_exterior.stop()
            self._pipe_wrist.stop()
        except Exception:
            pass
        log.info("[UR3e] Stopped.")


# ══════════════════════════════ Policy Loader ══════════════════════════════

def load_policy(checkpoint_path, device='cuda', resize_shape=None, num_inference_steps=16):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    shape_meta = payload['shape_meta']

    normalizer = LinearNormalizer()
    normalizer.load_state_dict(payload['normalizer_state_dict'])

    encoder = MultiImageObsEncoder(
        shape_meta, use_group_norm=True,
        resize_shape=resize_shape,
    )
    # DDIM at inference: 16 deterministic steps instead of 100 DDPM steps (~6x faster)
    scheduler = DDIMScheduler(num_train_timesteps=100)
    policy = DiffusionUnetImagePolicy(
        obs_encoder=encoder,
        noise_scheduler=scheduler,
        shape_meta=shape_meta,
        horizon=16,
        n_obs_steps=2,
        n_action_steps=8,
        num_inference_steps=num_inference_steps,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
    )
    policy.set_normalizer(normalizer)
    policy.load_state_dict(payload['policy_state_dict'])
    policy.to(device)
    policy.eval()
    return policy, shape_meta


def make_vis_frame(obs, action, step, inference_ms):
    """Side-by-side exterior + wrist camera view with status overlay."""
    ext = cv2.cvtColor(obs['exterior_rgb'], cv2.COLOR_RGB2BGR)
    wrist = cv2.cvtColor(obs['wrist_rgb'], cv2.COLOR_RGB2BGR)
    disp = 320
    frame = np.concatenate(
        [cv2.resize(ext, (disp, disp)), cv2.resize(wrist, (disp, disp))], axis=1)
    lines = [
        f'Step: {step}',
        f'Infer: {inference_ms:.0f}ms',
        f'Gripper: {"open" if action[6] < 0.5 else "close"}',
        "q / ESC — stop",
    ]
    for i, txt in enumerate(lines):
        cv2.putText(frame, txt, (10, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


# ══════════════════════════════ Inference Loop ══════════════════════════════

def build_obs_dict(obs_history, device='cuda'):
    """
    Convert raw observation history to policy input tensors.

    obs_history: deque of dicts from UR3eRobot.get_obs(), length = n_obs_steps
    Returns: obs_dict with (1, n_obs_steps, ...) tensors, keys matching training format.
    """
    exterior_stack = np.stack([o['exterior_rgb'] for o in obs_history])  # (T, H, W, 3)
    wrist_stack = np.stack([o['wrist_rgb'] for o in obs_history])
    state_stack = np.stack([o['state'] for o in obs_history])  # (T, state_dim)

    # HWC uint8 -> CHW float32 [0, 1]
    ext_t = torch.from_numpy(exterior_stack).float().permute(0, 3, 1, 2) / 255.0
    wrist_t = torch.from_numpy(wrist_stack).float().permute(0, 3, 1, 2) / 255.0
    state_t = torch.from_numpy(state_stack).float()

    return {
        'observation_images_exterior_image_1_left': ext_t.unsqueeze(0).to(device),
        'observation_images_wrist_image_left': wrist_t.unsqueeze(0).to(device),
        'observation_state': state_t.unsqueeze(0).to(device),
    }


def run_inference(policy, robot, device='cuda',
                  n_obs_steps=2, n_action_steps=8,
                  steps_per_inference=6,
                  max_steps=500, frequency=10,
                  dry_run=False):
    dt = 1.0 / frequency
    frame_latency = 1.0 / 30  # camera runs at 30 fps
    obs_history = deque(maxlen=n_obs_steps)

    # Fill obs history before first inference
    log.info("Warming up cameras...")
    for _ in range(n_obs_steps):
        obs_history.append(robot.get_obs())
        time.sleep(dt)

    # One warm-up inference pass to trigger CUDA kernel compilation
    log.info("Warming up policy inference...")
    with torch.no_grad():
        policy.predict_action(build_obs_dict(obs_history, device))
    log.info(f"Ready. freq={frequency}Hz  steps_per_inference={steps_per_inference}  "
             f"max_steps={max_steps}  dry_run={dry_run}")

    step = 0
    iter_idx = 0
    t_start = time.monotonic()

    try:
        while step < max_steps:
            # Deadline for this inference cycle
            t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

            obs_dict = build_obs_dict(obs_history, device)

            t_inf = time.monotonic()
            with torch.no_grad():
                action_chunk = policy.predict_action(obs_dict)
            inference_ms = (time.monotonic() - t_inf) * 1000

            actions = action_chunk[0].cpu().numpy()  # (n_action_steps, action_dim)
            log.info(f"step {step:>4}: inference={inference_ms:.0f}ms")

            n_exec = min(steps_per_inference, n_action_steps, max_steps - step)
            for i in range(n_exec):
                # Wait until this action's scheduled time slot
                precise_wait(t_start + (iter_idx + i) * dt)

                if not dry_run:
                    robot.apply_action(actions[i])

                obs = robot.get_obs()
                obs_history.append(obs)

                # Visualize — non-blocking, matches model input (same crop+resize)
                cv2.imshow('Diffusion Policy', make_vis_frame(obs, actions[i], step, inference_ms))
                key = cv2.pollKey()
                if key in (ord('q'), 27):  # q or ESC
                    raise KeyboardInterrupt

                step += 1

            # Wait until cycle end, leaving one camera frame of slack for next obs
            precise_wait(t_cycle_end - frame_latency)
            iter_idx += steps_per_inference

    except KeyboardInterrupt:
        log.info("Stopped by user.")
    finally:
        cv2.destroyAllWindows()
        if not dry_run:
            robot.stop()
        log.info(f"Inference ended at step {step}.")


# ══════════════════════════════ Main ══════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='UR3e real robot inference')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--robot_ip', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--frequency', type=int, default=10)
    parser.add_argument('--steps_per_inference', type=int, default=6,
                        help='Actions to execute per inference cycle (default: 6)')
    parser.add_argument('--num_inference_steps', type=int, default=16,
                        help='DDIM denoising steps at inference (default: 16)')
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--resize', type=int, nargs=2, default=None, metavar=('H', 'W'))
    parser.add_argument('--no_gripper', action='store_true')
    parser.add_argument('--dry_run', action='store_true',
                        help='Run full pipeline without sending robot commands')
    parser.add_argument('--cam_exterior', default=CAM_SERIAL_EXTERIOR,
                        help='RealSense serial number for exterior camera')
    parser.add_argument('--cam_wrist', default=CAM_SERIAL_WRIST,
                        help='RealSense serial number for wrist camera')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)

    log.info("Loading policy...")
    policy, shape_meta = load_policy(
        args.checkpoint, args.device,
        resize_shape=tuple(args.resize) if args.resize else None,
        num_inference_steps=args.num_inference_steps)
    log.info(f"Policy loaded. Action dim: {shape_meta['action']['shape'][0]}, "
             f"DDIM steps: {args.num_inference_steps}")

    log.info(f"Connecting to UR3e at {args.robot_ip}...")
    robot = UR3eRobot(
        robot_ip=args.robot_ip,
        frequency=args.frequency,
        use_gripper=not args.no_gripper,
        cam_serial_exterior=args.cam_exterior,
        cam_serial_wrist=args.cam_wrist,
    )

    robot.home()

    run_inference(
        policy=policy,
        robot=robot,
        device=args.device,
        frequency=args.frequency,
        steps_per_inference=args.steps_per_inference,
        max_steps=args.max_steps,
        dry_run=args.dry_run,
    )


if __name__ == '__main__':
    main()
