# Diffusion Policy — Zero-to-One Reimplementation Guide

Reference implementation: [Chi et al., 2023 (RSS)](https://diffusion-policy.cs.columbia.edu/)  
Source repo analyzed: `Imitation-Learning/diffusion_policy-main/`

---

## 1. Core Concepts

### 1.1 The Diffusion Policy Idea

Standard BC maps observation directly to action (unimodal, MSE regression):

```
π(obs) → action
```

Diffusion Policy instead learns the reverse of a noise-corruption process:

```
Forward:  action → noisy_action   (add Gaussian noise over T steps)
Reverse:  noisy_action → action   (learned denoiser conditioned on obs)
```

At inference: sample pure Gaussian noise, run 100 denoising steps conditioned on
the current observation → produces a clean action sequence.

**Why this works**: representing the action distribution as a reverse diffusion
process naturally captures **multimodal behavior** (e.g., going left or right around
an obstacle) that MSE regression or Gaussian policies collapse to the mean.

---

### 1.2 Temporal Structure (Action Chunking)

The policy reasons over a temporal window, not a single timestep:

```
Timeline:
  [obs_{t-1}, obs_t] → predict [a_t, a_{t+1}, ..., a_{t+7}]
   ↑ To=2 obs steps             ↑ Ta=8 action steps
   |←————————— Horizon H=16 ——————————————→|
```

- **Observation horizon** `To=2`: condition on last 2 image frames
- **Action horizon** `Ta=8`: predict 8 future actions at once
- **Execute** only the first `n_action_steps` actions before re-planning

This receding-horizon control reduces compounding errors vs. single-step prediction.

---

### 1.3 Two Network Backbones

| Backbone | Class | Conditioning mechanism |
|---|---|---|
| **1D U-Net** | `ConditionalUnet1D` | Global: concat obs emb + step emb via FiLM |
| **Transformer** | `TransformerForDiffusion` | Cross-attention over obs tokens |

### 1.4 Two Observation Modes

| Mode | Input | Vision encoder |
|---|---|---|
| **Lowdim** | State vector (joint angles, pos, etc.) | None |
| **Image / Hybrid** | RGB image + low-dim state | ResNet-18/34 (GroupNorm) |

---

## 2. Project Architecture

There are **two repos** involved. Keep them clearly separate in your mind:

| Repo | Purpose |
|---|---|
| `diffusion_policy-main/` | **Reference code** — the original paper implementation. Read it, learn from it, compare against it. Do not copy-paste from it. |
| `diffusion-policy-reimplementation/` | **Your code** — what you build from scratch, stage by stage. This is your portfolio project. |

---

### 2.1 Reference Repo (`diffusion_policy-main/`) — READ FROM HERE

```
diffusion_policy-main/
├── train.py / eval.py              ← Entry points (Hydra-driven)
├── diffusion_policy/
│   ├── config/                     ← All Hydra YAML configs
│   │   └── task/                   ← Per-task configs (pusht, kitchen, ...)
│   ├── policy/                     ← Policy classes (primary abstraction)
│   │   ├── diffusion_unet_lowdim_policy.py
│   │   ├── diffusion_unet_image_policy.py
│   │   ├── diffusion_unet_hybrid_image_policy.py   ← most common
│   │   ├── diffusion_transformer_lowdim_policy.py
│   │   └── diffusion_transformer_hybrid_image_policy.py
│   ├── model/
│   │   ├── diffusion/
│   │   │   ├── conditional_unet1d.py        ← Core 1D U-Net denoiser
│   │   │   ├── transformer_for_diffusion.py
│   │   │   ├── ema_model.py                 ← EMA weight averaging
│   │   │   └── mask_generator.py            ← Inpainting masks
│   │   └── vision/
│   │       └── multi_image_obs_encoder.py   ← Image feature extractor
│   ├── dataset/
│   │   ├── pusht_image_dataset.py           ← Dataset class
│   │   └── replay_buffer.py                 ← Zarr-based storage
│   ├── env/
│   │   └── pusht/pusht_image_env.py         ← PushT simulation
│   ├── env_runner/
│   │   └── pusht_image_runner.py            ← Vectorized eval
│   ├── workspace/
│   │   └── train_diffusion_unet_hybrid_workspace.py  ← Training orchestrator
│   └── common/
│       ├── normalizer.py                    ← LinearNormalizer
│       └── sampler.py                       ← SequenceSampler
```

**Design philosophy of the reference**: O(N+M) — adding a new task requires only
a new dataset + env_runner + config. Adding a new method requires only a new
policy + workspace. No cross-product of code.

---

### 2.2 Your Reimplementation Repo (`diffusion-policy-reimplementation/`) — BUILD HERE

Organized by learning stage, not by component type. Each stage is independently
runnable so you can verify it before moving on.

```
diffusion-policy-reimplementation/
├── stage1_ddpm/
│   ├── noise_scheduler.py      ← your DDPM math
│   ├── unet1d.py               ← your 1D U-Net
│   └── train_toy.py            ← toy validation (bimodal distribution test)
├── stage2_dataset/
│   ├── replay_buffer.py        ← Zarr reader
│   ├── sampler.py              ← sequence sampling with padding
│   └── normalizer.py           ← linear normalizer
├── stage3_lowdim/
│   ├── policy.py               ← DiffusionUnetLowdimPolicy
│   └── train.py                ← training loop (imports stage1 + stage2)
├── stage4_vision/
│   └── encoder.py              ← MultiImageObsEncoder (GroupNorm backbone)
├── stage5_image_policy/
│   └── policy.py               ← DiffusionUnetHybridImagePolicy
├── stage6_training/
│   ├── ema.py                  ← EMAModel
│   └── workspace.py            ← full training orchestrator
└── stage7_eval/
    ├── env_runner.py            ← vectorized rollout
    └── eval.py                  ← evaluation entry point
```

**How the two repos relate at each stage**:

| Your file | Read from reference |
|---|---|
| `stage1_ddpm/noise_scheduler.py` | `diffusion_policy/model/diffusion/` (schedulers) |
| `stage1_ddpm/unet1d.py` | `diffusion_policy/model/diffusion/conditional_unet1d.py` |
| `stage2_dataset/sampler.py` | `diffusion_policy/common/sampler.py` |
| `stage2_dataset/normalizer.py` | `diffusion_policy/common/normalizer.py` |
| `stage3_lowdim/policy.py` | `diffusion_policy/policy/diffusion_unet_lowdim_policy.py` |
| `stage4_vision/encoder.py` | `diffusion_policy/model/vision/multi_image_obs_encoder.py` |
| `stage5_image_policy/policy.py` | `diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py` |
| `stage6_training/ema.py` | `diffusion_policy/model/diffusion/ema_model.py` |
| `stage6_training/workspace.py` | `diffusion_policy/workspace/train_diffusion_unet_hybrid_workspace.py` |
| `stage7_eval/env_runner.py` | `diffusion_policy/env_runner/pusht_image_runner.py` |

---

## 3. End-to-End Data Flow

### Training

```
Zarr file (demonstration data)
  ↓  PushTImageDataset + SequenceSampler
  ↓  Batch: {image: [B, To, 3, 96, 96], agent_pos: [B, To, 2], action: [B, Ta, 2]}
  ↓  LinearNormalizer → scale all values to [-1, 1]
  ↓  MultiImageObsEncoder → obs_feature [B, D_obs]
  ↓  LowdimMaskGenerator → obs timesteps fixed, action timesteps noised
  ↓  DDPMScheduler.add_noise(action, ε, t) → x_t
  ↓  ConditionalUnet1D(x_t, t_emb, obs_cond) → ε_pred
  ↓  Loss = MSE(ε_pred, ε)
  ↓  AdamW step + EMA update
```

### Inference

```
obs = env.reset() → {image: [To, 3, 96, 96], agent_pos: [To, 2]}
  ↓  normalize obs
  ↓  MultiImageObsEncoder → obs_feature
  ↓  x_T = torch.randn([1, H, action_dim])        ← pure Gaussian noise
  ↓  for t in scheduler.timesteps (T → 0):
       ε_pred = ConditionalUnet1D(x_t, t, obs_cond)
       x_{t-1} = DDPMScheduler.step(ε_pred, t, x_t)
  ↓  x_0 = predicted clean action sequence [1, H, action_dim]
  ↓  unnormalize → execute first n_action_steps actions → re-plan
```

---

## 4. Stage-by-Stage Reimplementation

### Stage 1 — Core Diffusion Mechanics (no robot, no images)

Understand and reimplement DDPM from scratch.

**Reference file**: `diffusion_policy/model/diffusion/conditional_unet1d.py`

```python
# Step 1: Sinusoidal time embedding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim): ...
    def forward(self, t):   # t: [B] → [B, dim]
        ...

# Step 2: 1D ResNet block with FiLM conditioning
class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3): ...
    def forward(self, x, cond):
        # cond → linear → scale + shift → apply to x after conv
        ...

# Step 3: 1D U-Net denoiser
class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim, down_dims=[256, 512, 1024]): ...
    def forward(self, sample, timestep, global_cond=None):
        # sample:      [B, H, action_dim]  (noisy action sequence)
        # timestep:    [B]
        # global_cond: [B, obs_dim]
        ...

# Step 4: DDPM Scheduler (or use diffusers.DDPMScheduler directly)
# Key API:
#   scheduler.add_noise(x0, noise, t)   → x_t   (training)
#   scheduler.step(eps_pred, t, x_t)    → x_{t-1} (inference)
```

**Validation**: Train on a toy 2D multimodal dataset (e.g., bimodal Gaussian or
figure-8 trajectory). Confirm the model generates multimodal samples rather than
collapsing to the mean.

---

### Stage 2 — Dataset + Normalizer

**Reference files**:
- `diffusion_policy/dataset/pusht_image_dataset.py`
- `diffusion_policy/common/normalizer.py`
- `diffusion_policy/common/sampler.py`

**Zarr data format**:

```
pusht_cchi_v7_replay.zarr/
├── data/
│   ├── img       [N_total_steps, 96, 96, 3]    uint8
│   ├── action    [N_total_steps, 2]             float32
│   └── state     [N_total_steps, 5]             float32
└── meta/
    └── episode_ends  [N_episodes]               int64  (cumulative step counts)
```

**Key components**:

```python
class SequenceSampler:
    """Samples temporal windows of length H from episode-based data."""
    def __init__(self, replay_buffer, sequence_length, pad_before, pad_after):
        # Build index of all valid (episode, start_step) pairs
        ...
    def __getitem__(self, idx):
        # Returns {obs, action} window of length H
        # Pads at episode boundaries (repeat first/last frame)
        ...

class LinearNormalizer:
    """Fit to data statistics, normalize to [-1, 1]."""
    def fit(self, data):        # compute min/max (or mean/std)
    def normalize(self, x):
    def unnormalize(self, x):
```

**Important**: pad_before=1, pad_after=7 for PushT. This ensures the first and
last frames of each episode can be sampled as part of a valid window.

---

### Stage 3 — State-Only Policy (Lowdim)

Start without images to isolate the diffusion logic.

**Reference file**: `diffusion_policy/policy/diffusion_unet_lowdim_policy.py`

```python
class DiffusionUnetLowdimPolicy(nn.Module):
    def __init__(self, model, noise_scheduler, horizon, obs_dim, action_dim,
                 n_obs_steps, n_action_steps):
        self.model = model                  # ConditionalUnet1D
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()

    def set_normalizer(self, normalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # 1. Normalize: obs [B, To, obs_dim], action [B, H, action_dim]
        nobs    = self.normalizer['obs'].normalize(batch['obs'])
        naction = self.normalizer['action'].normalize(batch['action'])

        # 2. Flatten obs as global condition
        B = nobs.shape[0]
        global_cond = nobs.reshape(B, -1)   # [B, To * obs_dim]

        # 3. Sample noise and diffusion timestep
        noise = torch.randn_like(naction)
        t = torch.randint(0, self.noise_scheduler.num_train_timesteps, (B,))

        # 4. Add noise to action
        noisy_action = self.noise_scheduler.add_noise(naction, noise, t)

        # 5. Predict noise
        eps_pred = self.model(noisy_action, t, global_cond=global_cond)

        # 6. MSE loss
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def predict_action(self, obs_dict):
        # 1. Normalize and flatten obs
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B = nobs.shape[0]
        global_cond = nobs.reshape(B, -1)

        # 2. Start from pure noise
        x = torch.randn((B, self.horizon, self.action_dim), device=device)

        # 3. Denoise
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            eps_pred = self.model(x, t, global_cond=global_cond)
            x = self.noise_scheduler.step(eps_pred, t, x).prev_sample

        # 4. Unnormalize and return first n_action_steps
        action = self.normalizer['action'].unnormalize(x)
        return action[:, :self.n_action_steps]
```

---

### Stage 4 — Vision Encoder (GroupNorm, not BatchNorm)

**Reference file**: `diffusion_policy/model/vision/multi_image_obs_encoder.py`

```python
class MultiImageObsEncoder(nn.Module):
    """Encodes RGB images + low-dim observations into a flat feature vector."""

    def __init__(self, shape_meta, vision_backbone, feature_dim, ...):
        # Load ResNet-18/34 backbone
        # CRITICAL: replace ALL BatchNorm2d → GroupNorm(num_groups, channels)
        # BatchNorm does not work with EMA — EMA averages BN running statistics
        # incorrectly, causing train/eval distribution mismatch.
        ...

    def forward(self, obs_dict):
        # For each image key:
        #   [B, To, 3, H, W] → [B*To, 3, H, W] → CNN → [B*To, feat_dim]
        #   → reshape → [B, To * feat_dim]
        # Concatenate with low-dim obs: [B, To * lowdim_dim]
        # Return: [B, total_obs_dim]
        ...
```

**Why GroupNorm**: EMA maintains a running average of model weights. BatchNorm
has `running_mean` / `running_var` buffers that are updated via exponential
moving average of batch statistics during the forward pass — not via the
gradient-based EMA. This creates a mismatch. GroupNorm has no running statistics,
so it is fully compatible with weight-space EMA.

---

### Stage 5 — Full Image Policy (Hybrid)

**Reference file**: `diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py`

Combine Stage 3 (diffusion) + Stage 4 (vision encoder):

```python
class DiffusionUnetHybridImagePolicy(nn.Module):
    def __init__(self, shape_meta, noise_scheduler, horizon, ...):
        self.obs_encoder = MultiImageObsEncoder(...)   # CNN backbone
        self.model = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_encoder.output_dim,
            down_dims=[256, 512, 1024],
        )
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()

    def compute_loss(self, batch):
        # Same as lowdim, but:
        nobs_img = self.normalizer['image'].normalize(batch['image'])
        global_cond = self.obs_encoder({'image': nobs_img, 'agent_pos': ...})
        # ... rest identical to Stage 3

    def predict_action(self, obs_dict):
        # Same as lowdim, but encode images first:
        global_cond = self.obs_encoder(obs_dict)
        # ... rest identical to Stage 3
```

---

### Stage 6 — Training Loop + EMA

**Reference files**:
- `diffusion_policy/workspace/train_diffusion_unet_hybrid_workspace.py`
- `diffusion_policy/model/diffusion/ema_model.py`

```python
# EMA: maintain a shadow copy of policy weights for stable inference
ema_policy = copy.deepcopy(policy)
ema_model = EMAModel(ema_policy.parameters(), power=0.75, max_value=0.9999)

optimizer = AdamW(policy.parameters(), lr=1e-4, weight_decay=1e-6)
lr_scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps=500)

global_step = 0
for epoch in range(num_epochs):   # ~3050 epochs for PushT
    for batch in dataloader:      # batch_size=64
        loss = policy.compute_loss(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

        # EMA update: decay = 1 - (1 + step/gamma)^(-power)
        ema_model.step(policy.parameters())
        global_step += 1

    if epoch % eval_every == 0:   # every 50 epochs
        # Use ema_policy (NOT policy) for rollouts and validation
        ema_model.copy_to(ema_policy.parameters())
        scores = env_runner.run(ema_policy)
        val_loss = compute_val_loss(ema_policy, val_dataloader)
        save_checkpoint(epoch, policy, ema_policy, optimizer, scores)
```

**Key hyperparameters**:

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-6 |
| LR schedule | Cosine with 500-step warmup |
| Batch size | 64 |
| EMA power | 0.75 |
| EMA max_value | 0.9999 |
| DDPM timesteps (train) | 100 |
| DDIM timesteps (eval) | 8–16 (10× speedup) |
| Num epochs (PushT) | ~3050 |
| Eval every | 50 epochs |

---

### Stage 7 — Environment + Evaluation

**Reference files**:
- `diffusion_policy/env/pusht/pusht_image_env.py`
- `diffusion_policy/env_runner/pusht_image_runner.py`

**PushT task**:
- 2D simulation (PyGame + Pymunk physics)
- Goal: push a T-shaped block to the target pose
- State: `agent_pos (2), block_pos (2), block_angle (1)`
- Action: `agent_goal_pos (2)` — position controller
- Image: 96×96 RGB render

**Eval runner**:

```python
class PushTImageRunner:
    def run(self, policy):
        # 56 parallel episodes via AsyncVectorEnv (6 train + 50 test)
        # Each episode: max 300 steps, fps=10
        # Metric: max_reward per episode (1.0 = block reached target)
        # Aggregates: mean_score, per-seed scores
        # Logs videos to WandB for first 6 episodes
        ...
```

Target performance: **>0.70 mean_score** on the 50 test episodes.

---

## 5. Minimal Reimplementation Checklist

```
Stage 1 — Core diffusion
  □ SinusoidalPosEmb
  □ ConditionalResidualBlock1D (FiLM conditioning)
  □ ConditionalUnet1D (encoder-decoder with skip connections)
  □ DDPM forward + reverse process
  □ Validate on 2D multimodal toy dataset

Stage 2 — Data pipeline
  □ Zarr dataset reader
  □ SequenceSampler with episode boundary padding
  □ LinearNormalizer (fit + normalize + unnormalize)
  □ Verify temporal windowing with shape assertions

Stage 3 — Lowdim policy
  □ DiffusionUnetLowdimPolicy.compute_loss
  □ DiffusionUnetLowdimPolicy.predict_action
  □ Train on PushT state data → confirm loss converges

Stage 4 — Vision encoder
  □ ResNet backbone with GroupNorm (not BatchNorm)
  □ MultiImageObsEncoder (temporal stacking + concat low-dim)
  □ Verify output shapes match ConditionalUnet1D global_cond_dim

Stage 5 — Image policy
  □ DiffusionUnetHybridImagePolicy (connect stages 3 + 4)
  □ Confirm compute_loss and predict_action run end-to-end

Stage 6 — Training loop
  □ EMAModel (shadow copy, decay schedule)
  □ AdamW + cosine LR with warmup
  □ Checkpoint saving (top-k by mean_score)
  □ WandB logging

Stage 7 — Evaluation
  □ PushTImageEnv (or use original)
  □ Vectorized rollout runner
  □ mean_score metric
  □ Target: >0.70 on 50 test episodes
```

---

## 6. Key Papers

| Paper | Role |
|---|---|
| Ho et al. 2020 — DDPM | Foundation: forward/reverse diffusion process |
| Song et al. 2020 — DDIM | Fast inference (8–16 steps vs 100) |
| Chi et al. 2023 — Diffusion Policy | The paper itself |
| Janner et al. 2022 — Diffuser | Related: diffusion over full trajectories |

---

## 7. Common Pitfalls

1. **BatchNorm + EMA**: Always replace BatchNorm with GroupNorm in any backbone
   used under EMA. Failing to do this causes silent degradation at eval time.

2. **Normalizer scope**: Fit the normalizer on training data only, before training
   starts. Set it in the policy via `set_normalizer()`. Never refit during eval.

3. **Action inpainting vs. pure generation**: The mask generator fixes observation
   timesteps and samples over action timesteps only. Confusing these will produce
   degenerate outputs.

4. **n_action_steps < horizon**: Only execute the first `n_action_steps` from each
   prediction window. Re-plan at every step. Never execute all H predicted actions.

5. **DDIM at eval time**: Switch from DDPM (100 steps) to DDIM (8–16 steps) only
   at inference. Training always uses the full DDPM schedule.

6. **Padding at episode boundaries**: `pad_before=1, pad_after=7` ensures edge
   frames can be sampled. Missing this causes shape errors or biased sampling.
