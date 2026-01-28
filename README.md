# Diffusion Policy Reimplementation

A from-scratch reimplementation of [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) for robot learning, focusing on visuomotor policy learning via conditional action diffusion.

## Demo

<!-- TODO: Add demo GIF/video here -->

---

## 1. Motivation

This is a **from-scratch reimplementation** of Diffusion Policy, not a fork. Every line of code is written by hand after studying the original paper and codebase. The primary goal is to deeply understand how diffusion models work in robot action space; the secondary goal is to provide a clean, minimal codebase others can learn from.

For a step-by-step learning guide, see:
- [reimplementation-guide.md](reimplementation-guide.md) — core concepts and architecture analysis
- [how-to-code.md](how-to-code.md) — hands-on coding walkthrough for each component

### Differences from the Original

| Aspect | [Original (Chi et al.)](https://github.com/real-stanford/diffusion_policy) | This Project |
|--------|----------------------------------------------------------------------------|--------------|
| **Code origin** | Complete framework (~20k LOC) | From-scratch rewrite (~2k LOC) |
| **Config system** | Hydra + YAML (dozens of config files) | Plain argparse (one `train.py`) |
| **Data format** | Zarr only | Zarr + LeRobot v3 (modern standard) |
| **Wandb** | Deeply coupled, required | Optional (`--wandb` flag) |
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

#### 2.1.3 Two Network Backbones

| Backbone | Class | Conditioning Mechanism |
| --- | --- | --- |
| **1D U-Net** | `ConditionalUnet1D` | Global: concat obs_emb + step_emb via FiLM |
| **Transformer** | `TransformerForDiffusion` | Cross-attention over obs tokens |

This project implements the **1D U-Net** backbone.

#### 2.1.4 Two Observation Modes
| Mode | Input | Vision Encoder |
| --- | --- | --- |
| **Lowdim** | State vector (joint angles, pos, etc.) | None |
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
│   │   ├── lerobot_wrapper.py      # LeRobot dataset wrapper for image policy
│   │   ├── replay_buffer.py        # Zarr-based replay buffer for lowdim policy
│   │   └── sampler.py              # Sliding window sequence sampler with padding
│   └── policy/
│       ├── image.py                # Image-conditioned diffusion policy
│       └── lowdim.py               # Low-dim observation diffusion policy
├── training/
│   ├── workspace.py                # Lowdim training workspace (zarr pipeline)
│   └── workspace_image.py          # Image training workspace (LeRobot pipeline)
├── eval/
│   └── runner.py                   # Evaluation runner with action chunking loop
├── scripts/
│   └── hdf5_to_lerobot.py          # HDF5 → LeRobot v3 format converter
├── train.py                        # Training entry point (--mode lowdim/image)
├── eval.py                         # Evaluation entry point
├── experiments/                    # Stage-by-stage validation scripts
│   ├── 01_ddpm_toy.py              # DDPM on toy 1D data
│   ├── 02_dataset_test.py          # Zarr dataset pipeline test
│   ├── 03_lowdim_train.py          # Lowdim policy training test
│   ├── 04_encoder_test.py          # Vision encoder unit test
│   ├── 05_image_policy_test.py     # Image policy integration test
│   ├── 06_train_smoke.py           # Full training smoke test
│   └── 07_eval_test.py             # Evaluation pipeline test
├── reimplementation-guide.md       # Core concepts and architecture deep-dive
├── how-to-code.md                  # Hands-on coding guide for each stage
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

### Training (Image Policy, LeRobot format)
```bash
python train.py --mode image \
    --repo_id local/my_dataset --root ./data/my_dataset \
    --epochs 3000 --batch 8 --resize 224 224 \
    --device cuda --output_dir outputs/my_run
```

Wandb logging is enabled by default (project: `Diffusion-Policy`). Use `--no_wandb` to disable, or `--wandb_run_name` to set a custom run name.

### Training (Lowdim Policy, zarr format)
```bash
python train.py --mode lowdim \
    --data data/demo.zarr \
    --epochs 3000 --device cuda
```

### Evaluation
```bash
python eval.py --checkpoint outputs/policy_final.pt --mock
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