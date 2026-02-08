# Diffusion Policy Reimplementation

A from-scratch reimplementation of [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) for robot learning, focusing on visuomotor policy learning via conditional action diffusion.

## Demo

<!-- TODO: Add demo GIF/video here -->

---

## TODO

- [ ] **Demo** — Add demo GIF / video of real robot pick-and-place deployment
- [ ] **Checkpoints** — Release trained checkpoints with download link
- [x] **Observation timestamp alignment** — Fixed: obs uses `[-(To-1)/fps, ..., 0]`, action uses `[-(To-1)/fps, ..., (H-To)/fps]`. Training obs now covers past frames `[t-To+1, ..., t]`, matching inference obs_deque.
- [x] **Action extraction offset** — Fixed: `start = To - 1` (execute from current time t). Consistent with original paper convention and the corrected timestamp alignment.
- [ ] **Real environment evaluation** — `eval.py` currently only supports `--mock` mode with random observations. A proper evaluation requires a gym-compatible simulated environment (e.g., PushT, RoboSuite) or real robot rollout.

---

## 1. Motivation

This is a **from-scratch reimplementation** of Diffusion Policy, not a fork. Every line of code is written by hand after studying the original paper and codebase. The primary goal is to deeply understand how diffusion models work in robot action space; the secondary goal is to provide a clean, minimal codebase others can learn from.

### Differences from the Original

| Aspect | [Original (Chi et al.)](https://github.com/real-stanford/diffusion_policy) | This Project |
|--------|----------------------------------------------------------------------------|--------------|
| **Code origin** | Complete framework (~20k LOC) | From-scratch rewrite (~2k LOC) |
| **Config system** | Hydra + YAML (dozens of config files) | Plain argparse (one `train.py`) |
| **Data format** | Zarr only | LeRobot v3 (modern standard) |
| **Wandb** | Deeply coupled, required | Always enabled (project: `Diffusion-Policy`) |
| **State normalization** | Action only | Action + state (inspired by OpenPI) |
| **Comments** | Minimal | Learning-note style: explains the *why* behind each design |
| **Environment** | conda + pip + manual setup | `bash setup.sh` one-click (uv) |
| **Scope** | PushT, RoboMimic, multiple backbones | Pick-and-Place with 1D U-Net (focused) |

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
│   │   │   ├── scheduler.py        # DDPM forward/reverse process (cosine schedule)
│   │   │   ├── unet1d.py           # Conditional 1D U-Net with FiLM conditioning
│   │   │   └── ema.py              # Exponential Moving Average for stable inference
│   │   └── vision/
│   │       └── encoder.py          # Multi-image observation encoder (ResNet18 + GroupNorm)
│   ├── dataset/
│   │   ├── normalizer.py           # Linear normalizer (limits/gaussian modes)
│   │   └── lerobot_wrapper.py      # LeRobot dataset wrapper for image policy
│   └── policy/
│       └── image.py                # Image-conditioned diffusion policy
├── training/
│   └── workspace.py                # Training workspace (LeRobot pipeline)
├── evaluation/
│   └── runner.py                   # Evaluation runner with action chunking loop
├── tests/
│   └── test_integration.py         # End-to-end pipeline smoke test
├── scripts/
│   └── hdf5_to_lerobot.py          # HDF5 → LeRobot v3 format converter
├── experiments/                     # Validation scripts
│   ├── 01_ddpm_toy.py              # DDPM on toy 1D data
│   ├── 02_encoder_test.py          # Vision encoder unit test
│   └── 03_image_policy_test.py     # Image policy integration test
├── train.py                        # Training entry point
├── inference.py                    # Real robot inference (UR3e + dual camera)
├── eval.py                         # Evaluation entry point (mock env)
├── setup.sh                        # One-click environment setup (uv + DP venv)
└── pyproject.toml                  # Dependencies and project metadata
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
```bash
python train.py \
    --repo_id local/my_dataset --root ./data/my_dataset \
    --epochs 3000 --batch 8 \
    --resize 96 96 --crop 76 76 \
    --device cuda --output_dir outputs/my_run
```

- `--resize H W` — resize images to H×W before the encoder (saves GPU memory)
- `--crop H W` — random crop to H×W during training, center crop at eval (data augmentation)

Wandb logging is always enabled (project: `Diffusion-Policy`). Use `--wandb_run_name` to set a custom run name.
Checkpoints are saved every `--save_every_steps` gradient steps (default: 5000) and every `--checkpoint_every` epochs (default: 100).

### Resume Training
```bash
python train.py \
    --repo_id local/my_dataset --root ./data/my_dataset \
    --resume outputs/my_run/checkpoint_epoch500.pt \
    --epochs 3000 --device cuda
```

### Real Robot Inference (UR3e)

Hardware: UR3e + Weiss CRG 30-050 gripper + dual Intel RealSense cameras.

```bash
# Install robot dependencies
pip install ur-rtde pyrealsense2 pyserial opencv-python

# Dry run (verify full pipeline without sending robot commands)
python inference.py --checkpoint outputs/policy_final.pt \
    --robot_ip 10.0.0.1 --dry_run

# Real deployment
python inference.py --checkpoint outputs/policy_final.pt \
    --robot_ip 10.0.0.1 \
    --frequency 10 --steps_per_inference 6 --num_inference_steps 16 \
    --max_steps 500

# Without gripper / custom camera serials
python inference.py --checkpoint outputs/policy_final.pt \
    --robot_ip 10.0.0.1 --no_gripper \
    --cam_exterior 105422061000 --cam_wrist 352122273671
```

- `--steps_per_inference` — actions to execute per inference cycle (default: 6)
- `--num_inference_steps` — DDIM denoising steps (default: 16, ~6× faster than DDPM-100)

### Evaluation (Mock)
```bash
python eval.py --checkpoint outputs/policy_final.pt --device cuda --mock
```

### Data Conversion (HDF5 → LeRobot)
```bash
python scripts/hdf5_to_lerobot.py \
    --input /path/to/hdf5_episodes \
    --repo_id local/my_dataset --root ./data/my_dataset \
    --fps 20 --state_keys qpos
```

---

## Acknowledgement

Reference: [Chi et al., 2023 (RSS)](https://diffusion-policy.cs.columbia.edu/)

Source repo: https://github.com/real-stanford/diffusion_policy
