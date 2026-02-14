# Diffusion Policy Reimplementation

A from-scratch reimplementation of [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) for robot learning, focusing on visuomotor policy learning via conditional action diffusion.

## Demo

<!-- TODO: add demo GIF — python scripts/summarize_trials.py --gif trials/<trial> --out docs/demo.gif -->

---

## TODO

- [ ] **Demo** — Add demo GIF / video of real robot pick-and-place deployment
- [ ] **Checkpoints** — Release trained checkpoints with download link
- [x] **Offline evaluation** — `eval.py` scores predicted vs recorded action chunks on held-out episodes; `train.py --val_ratio` logs the same metrics to wandb during training
- [ ] **Real-robot success rate** — run N trials with `inference.py --record_dir` and fill in [Results](#results) with `scripts/summarize_trials.py` (offline action error is a regression check, not a success metric)

Tooling for all three is in place (see [Recording Trials](#recording-trials-demo--success-rate)); what remains is training a model and running the trials on the robot.

---

## 1. Motivation

This is a **from-scratch reimplementation** of Diffusion Policy, not a fork. Every line of code is written by hand after studying the original paper and codebase. The primary goal is to deeply understand how diffusion models work in robot action space; the secondary goal is to provide a clean, minimal codebase others can learn from.

### Differences from the Original

| Aspect | [Original (Chi et al.)](https://github.com/real-stanford/diffusion_policy) | This Project |
|--------|----------------------------------------------------------------------------|--------------|
| **Code origin** | Complete framework (~20k LOC) | From-scratch rewrite (~5k LOC incl. tests) |
| **Config system** | Hydra + YAML (dozens of config files) | Plain argparse (one `train.py`) |
| **Data format** | Zarr only | LeRobot v3 (modern standard) |
| **State normalization** | Action only | Action + state (inspired by OpenPI) |
| **Comments** | Minimal | Learning-note style: explains the *why* behind each design |
| **Environment** | conda + pip + manual setup | `bash setup.sh` one-click (uv) |

---

## 2. Diffusion Policy

Diffusion Policy is a robot learning framework from [Chi et al., 2023 (RSS)](https://diffusion-policy.cs.columbia.edu/) that applies Denoising Diffusion Probabilistic Models (DDPM) to imitation learning. Instead of directly regressing actions, the policy learns to denoise random noise into a sequence of future actions conditioned on observations.

The key insight: representing the action distribution as a learned reverse diffusion process captures multimodal behavior (e.g., going left or right around an obstacle) far better than MSE regression or Gaussian policies.

---

### 2.1 Core Concepts

#### 2.1.1 The Diffusion Policy Idea

Standard BC maps observation directly to action (unimodal, MSE regression):

```
π(obs) → action
```

Diffusion Policy instead learns the reverse of a noise-corruption process:

```
Forward: action → noisy_action  (add Gaussian noise over T steps)
Reverse: noisy_action → action  (learned denoiser conditioned on obs)
```
At inference: sample pure Gaussian noise, run 100 denoising steps conditioned on the current observation → produces a clean action sequence.

**Why this works**: representing the action distribution as a reverse diffusion process naturally captures **multimodal behavior** (e.g., going left or right around an obstacle) that MSE regression or Gaussian policies collapse to the mean.

---

#### 2.1.2 Temporal Structure (Action Chunking)
The policy reasons over a temporal window, not a single timestep:

```
Timeline:
    [obs_{t-1}, obs_t] → PREDICT [a_t, a_{t+1}, ..., a_{t+7}]
     ↑ To=2 obs steps              ↑ Ta=8 action steps
     |<------------- Horizon H=16 ------------->|
```

- **Observation horizon** `To=2`: condition on last 2 image frames
- **Action horizon** `Ta=8`: predict 8 future actions at once
- **Execute** only the first `n_action_steps` actions before re-planning

This receding-horizon control reduces compounding errors vs. single-step prediction.

---

#### 2.1.3 Network Backbone

| Backbone | Class | Conditioning Mechanism |
| --- | --- | --- |
| **1D U-Net** | `ConditionalUnet1D` | Global: concat obs_emb + step_emb via FiLM |

The original paper also supports a Transformer backbone with cross-attention. This project implements the **1D U-Net** backbone.

#### 2.1.4 Observation Mode

| Mode | Input | Vision Encoder |
| --- | --- | --- |
| **Image / Hybrid** | RGB image + low-dim state | ResNet-18 (GroupNorm) |

---

## 3. Project Architecture
```
.
├── diffusion_policy/
│   ├── model/
│   │   ├── diffusion/
│   │   │   ├── scheduler.py        # DDPM / DDIM schedulers (cosine schedule)
│   │   │   ├── unet1d.py           # Conditional 1D U-Net with FiLM conditioning
│   │   │   └── ema.py              # Exponential Moving Average for stable inference
│   │   └── vision/
│   │       └── encoder.py          # Multi-image observation encoder (ResNet18 + GroupNorm)
│   ├── dataset/
│   │   ├── normalizer.py           # Linear normalizer (limits/gaussian modes)
│   │   └── lerobot_wrapper.py      # LeRobot dataset wrapper for image policy
│   ├── policy/
│   │   ├── image.py                # Image-conditioned diffusion policy
│   │   └── loader.py               # Build policy from config, load checkpoints (EMA-aware)
│   └── control.py                  # Real-robot scheduling: latency-aware chunk execution, gripper hysteresis
├── training/
│   └── workspace.py                # Training loop (LeRobot pipeline, EMA, val split, checkpoints)
├── evaluation/
│   ├── offline.py                  # Predicted vs recorded action error on held-out episodes
│   ├── runner.py                   # Rollout runner with action chunking loop (mock env)
│   ├── sim_policy.py               # Adapter for the shared Franka simulation (../robot-sim)
│   └── trial_recorder.py           # Save real-robot trials (videos, actions, outcome)
├── tests/                          # pytest: scheduler, normalizer, EMA, checkpoints, control, integration
├── scripts/
│   ├── hdf5_to_lerobot.py          # HDF5 → LeRobot v3 format converter
│   ├── visualize_dataset.py        # Inspect episodes in a LeRobot dataset
│   ├── benchmark_inference.py      # DDPM vs DDIM inference latency benchmark
│   └── summarize_trials.py         # Success rate (with 95% CI) + demo GIF from recorded trials
├── experiments/                     # Validation scripts
│   ├── 01_ddpm_toy.py              # DDPM on toy 1D data
│   ├── 02_encoder_test.py          # Vision encoder unit test
│   └── 03_image_policy_test.py     # Image policy integration test
├── train.py                        # Training entry point
├── inference.py                    # Real robot inference (UR3e + dual camera)
├── eval.py                         # Evaluation entry point (offline / mock)
├── setup.sh                        # One-click environment setup (uv + DP venv)
└── pyproject.toml                  # Dependencies, ruff config, project metadata
```

---

## 4. Setup

```bash
# One-click install (creates "DP" virtual environment via uv)
bash setup.sh

# Activate
source DP/bin/activate
```

Requires Python 3.10+ and a CUDA-capable GPU for training.

---

## 5. Usage

### Training

Standard format: **480×480** images, no resize, batch 64. Dataset images and inference camera frames must both be center-cropped to 480×480 square before conversion.

```bash
python train.py \
    --repo_id local/my_dataset --root ./data/my_dataset \
    --max_steps 30000 --batch 64 \
    --val_ratio 0.05 \
    --device cuda --output_dir outputs/my_run \
    --save_every_steps 5000 --max_keep_checkpoints 5 \
    --wandb_run_name my_run
```

- `--max_steps 30000` — total gradient steps; LR warms up linearly for `--lr_warmup_steps` (default 500), then cosine-decays to lr/10 at exactly this step
- `--val_ratio 0.05` — hold out 5% of episodes (split by episode); every `--val_every_steps` (default 2000) the EMA model is scored on them (`val/loss`, `val/action_mse`, `val/action_mae`). Default 0 = train on all episodes
- `--save_every_steps 5000` — save a step checkpoint every 5000 gradient steps; keep only the latest `--max_keep_checkpoints` (default: 3) to save disk space

`config.json` is written to `output_dir`, and the same config is stored **inside every checkpoint** — `eval.py` / `inference.py` rebuild the architecture from it, so non-default `--horizon`, `--crop`, etc. need no extra flags at load time.

Checkpoint files:
- `checkpoint_step*.pt` — raw training weights + EMA weights + optimizer state (for resuming; inference automatically uses the EMA weights)
- `policy_final.pt` — EMA weights only (for deployment)

### Resume Training
```bash
python train.py \
    --repo_id local/my_dataset --root ./data/my_dataset \
    --resume outputs/my_run/checkpoint_step20000.pt \
    --max_steps 30000 --val_ratio 0.05 --device cuda --output_dir outputs/my_run
```

Resuming continues the same wandb run. Pass the same `--val_ratio` / `--seed` as the original run so the train/val split is identical.

### Evaluation

Offline (default): compares the executed action chunk `action[To-1 : To-1+Ta]` with the recorded demonstration. Uses the val episodes stored in the checkpoint unless `--episodes` is given.

```bash
python eval.py --checkpoint outputs/my_run/policy_final.pt --device cuda
python eval.py --checkpoint outputs/my_run/policy_final.pt --device cuda --episodes 3 17 42 \
    --output_json outputs/my_run/eval.json
```

Reported: denoising loss, action MSE, action MAE (overall and per dim, in raw units — degrees for UR3e joints). Low action error does not guarantee task success (multimodal demos), so use it to compare checkpoints, not as a success rate.

Simulation (closed loop, success rate): see [Simulation (Franka Panda)](#simulation-franka-panda).

Mock rollout (pipeline smoke test only, reward is meaningless):
```bash
python eval.py --checkpoint outputs/my_run/policy_final.pt --mock
```

### Simulation (Franka Panda)

While the real robot is unavailable, policies are trained and evaluated in the shared
simulation repo [`../robot-sim`](../robot-sim) (robosuite Franka Panda + robomimic demos,
converted to LeRobot v3). The ACT reimplementation uses the same environment and seeds,
so success rates are directly comparable.

```bash
# one-time: install robot-sim into this environment and build a dataset there
uv pip install --python DP/bin/python -e ../robot-sim
(cd ../robot-sim && .venv/bin/python -m robot_sim.convert --task lift --root data/lerobot/lift_ph_abs_84)

# train (84×84 images, random crop 76×76, absolute 10-dim actions)
python train.py --repo_id local/robomimic_lift_ph_abs_84 --root ../robot-sim/data/lerobot/lift_ph_abs_84 \
    --max_steps 20000 --batch 64 --crop 76 76 --output_dir outputs/sim_lift

# closed-loop evaluation: 50 seeded episodes, videos of the first 5
python eval.py --checkpoint outputs/sim_lift/checkpoint_step5000.pt --device cuda \
    --sim lift --n_episodes 50 --video_dir outputs/sim_lift/videos
```

| Task | Steps | Success (seeds 100000–100019) |
|---|---|---|
| lift | 5k | 20/20 (95% CI 84–100%) |
| can | — | training |

`robot-sim` is not a declared dependency (CI has no `../robot-sim` checkout): the sim
tests skip when it is not installed, and a later `uv sync` removes it again.

### Real Robot Inference (UR3e)

Hardware: UR3e + Weiss CRG 30-050 gripper + dual Intel RealSense cameras.

```bash
# Install robot dependencies
pip install ur-rtde pyrealsense2 pyserial opencv-python

# Dry run (full pipeline, robot never moves — not even to home)
python inference.py --checkpoint outputs/policy_final.pt \
    --robot_ip 10.0.0.1 --dry_run

# Real deployment (control rate defaults to the training dataset's fps)
python inference.py --checkpoint outputs/policy_final.pt \
    --robot_ip 10.0.0.1 \
    --steps_per_inference 6 --num_inference_steps 16 \
    --max_steps 500
```

- `--frequency` — control rate in Hz; defaults to the dataset fps stored in the checkpoint and refuses a mismatch unless `--allow_fps_mismatch` is given
- `--steps_per_inference` — actions executed per inference cycle (default: 6 of the 8 predicted)
- `--num_inference_steps` — DDIM denoising steps (default: 16, ~6× faster than DDPM-100)
- `--max_steps` — control steps per trial (default: 800 = 40 s; the longest demo is ~33 s)
- `--max_joint_vel` — per-joint velocity clamp in rad/s (default: 1.0)
- `--record_dir` — record the trial (see below)
- `--no_gripper`, `--cam_exterior`, `--cam_wrist` — run without gripper / custom camera serials

Latency compensation: action slots that elapsed during inference are skipped, and execution always stops at the cycle boundary so delays never accumulate across cycles. If inference takes longer than a whole cycle, a warning is logged and the schedule is re-anchored.

Sensors are read by background threads (one per camera, one for the gripper's serial stream), so `get_obs()` only copies the latest cached readings and never blocks the control loop; it refuses data older than 0.5 s. Each cycle logs `infer`, `obs_max` (get_obs time), `late_max` (how late an action was sent) and `age_max` (sensor age), and the run ends with the effective control rate and a timing summary. **Check these in a `--dry_run` before the first real run**: `obs_max` should stay well below the 50 ms period and the effective rate should match the dataset fps.

Gripper convention: the training data uses **1 = open, 0 = closed** (every demonstration starts and ends open). Both the gripper state fed to the policy and the gripper action channel follow it.

### Recording Trials (Demo & Success Rate)

```bash
# Each run is saved to trials/trial_<timestamp>/ (exterior.mp4, wrist.mp4, steps.npz, meta.json);
# at the end you are asked whether the trial succeeded
python inference.py --checkpoint outputs/my_run/policy_final.pt --robot_ip 10.0.0.1 \
    --record_dir trials/

# Success rate per checkpoint with a 95% Wilson interval (dry runs and unanswered trials are skipped)
python scripts/summarize_trials.py --record_dir trials/ --markdown trials/results.md

# Side-by-side exterior | wrist demo GIF from one trial
python scripts/summarize_trials.py --gif trials/trial_20260212_101500 --out demo.gif
```

### Data Conversion (HDF5 → LeRobot)
```bash
python scripts/hdf5_to_lerobot.py \
    --input /path/to/hdf5_episodes \
    --repo_id local/my_dataset --root ./data/my_dataset \
    --fps 20 --state_keys qpos
```

---

## Results

<!-- Fill in from `python scripts/summarize_trials.py --record_dir trials/ --markdown trials/results.md` -->

| Checkpoint | Success | Rate | 95% CI | Mean duration |
|---|---|---|---|---|
| — | — | — | — | — |

---

## Acknowledgement

Reference: [Chi et al., 2023 (RSS)](https://diffusion-policy.cs.columbia.edu/)

Source repo: https://github.com/real-stanford/diffusion_policy
