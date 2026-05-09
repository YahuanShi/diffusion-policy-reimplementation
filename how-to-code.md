# How to Code This Yourself — Hands-On Guide

**Goal**: improve coding ability, master diffusion policy, build a portfolio for AI/robotics jobs.  
**Principle**: understand the math → write it yourself → validate → then read the reference.

> **Can I run this end-to-end after finishing all stages?**  
> Yes — but only after completing Stage 0 (environment + data) and Stage 8 (wiring
> all stages into one runnable system). The stages in between teach each component
> in isolation. Stage 8 is what makes it a real runnable project.

---

## Stage 0 — Environment Setup (do this first)

We use `uv` — the modern Python package manager (same team as `ruff`). It is
10–100× faster than pip/conda and generates a `uv.lock` lockfile for exact
reproducibility. No conda needed.

### 0A. Install uv (one-time, system-wide)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Restart your shell, then verify:
uv --version
```

### 0B. Set Up the Project

```bash
cd diffusion-policy-reimplementation

# Pin Python version
uv python pin 3.9

# Initialise the project (creates pyproject.toml)
uv init --no-readme
```

Replace the generated `pyproject.toml` with this:

```toml
[project]
name = "diffusion-policy-reimplement"
version = "0.1.0"
requires-python = "==3.9.*"
dependencies = [
    "torch",
    "torchvision",
    "diffusers",          # DDPM/DDIM scheduler reference (Stage 1)
    "zarr",               # dataset storage (Stage 2)
    "numpy",
    "transformers",       # get_cosine_schedule_with_warmup (Stage 6)
    "wandb",              # experiment tracking (Stage 6)
    "pymunk",             # PushT physics simulation (Stage 7)
    "pygame",             # PushT rendering (Stage 7)
    "matplotlib",         # toy validation plots (Stage 1)
    "einops",             # tensor reshaping utility
    "opencv-python",      # video logging
]

[tool.uv.sources]
torch = [
    { index = "pytorch-cu118", marker = "platform_system == 'Linux'" },
]
torchvision = [
    { index = "pytorch-cu118", marker = "platform_system == 'Linux'" },
]

[[tool.uv.index]]
name = "pytorch-cu118"
url = "https://download.pytorch.org/whl/cu118"
explicit = true

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = [
    "diffusion_policy",
    "training",
    "eval",
]
```

### 0C. Install All Dependencies

```bash
# Install everything + your own package in editable mode
uv sync
uv pip install -e .

# Generates uv.lock — commit this file to git for reproducibility
git add uv.lock pyproject.toml
```

From now on, run all scripts with `uv run`:
```bash
uv run python train.py
uv run python test_integration.py
uv run python experiments/01_ddpm_toy.py
```

Or activate the managed venv once per shell session:
```bash
source .venv/bin/activate
python train.py   # then use python directly
```

### 0D. Set Up the Package Structure

The `__init__.py` files and directory structure are already in the repo.
After `uv pip install -e .`, all imports work immediately:

```python
from diffusion_policy.model.diffusion.scheduler import DDPMScheduler
from diffusion_policy.model.diffusion.unet1d import ConditionalUnet1D
from diffusion_policy.dataset.normalizer import LinearNormalizer
# etc.
```

### 0E. Download the PushT Dataset

```bash
mkdir -p data
wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
unzip pusht.zip -d data/
# Result: data/pusht_cchi_v7_replay.zarr/
rm pusht.zip
```

Add to `.gitignore` so the dataset is never committed:
```
data/
.venv/
__pycache__/
*.pyc
wandb/
checkpoints/
```

### 0F. Reuse the PushT Environment (don't reimplement)

The PushT simulation (pygame + pymunk physics) is infrastructure, not a learning
objective. Reuse it directly from the reference repo:

```bash
uv pip install -e ../diffusion_policy-main
```

Then in your Stage 7 eval code:
```python
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
```

This is the only import from the reference. Everything else you write yourself.

### 0G. Verify the Setup

```bash
uv run python -c "
import torch
print('torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
import zarr, diffusers, wandb, pymunk, pygame
print('All imports OK')
"
```

All lines should print without error before starting Stage 1.

---

## Overall Learning Strategy

### The Three-Pass Rule

For every stage, do three passes in order. Never skip to pass 3.

```
Pass 1 — Understand (read, draw, derive)
  Read the relevant paper section or math.
  Draw the data shapes on paper before writing any code.
  Understand WHY each component exists, not just what it does.

Pass 2 — Implement yourself (blank file, no reference)
  Open a blank .py file.
  Write the component from memory / from math only.
  It will be wrong or incomplete. That is fine and expected.

Pass 3 — Compare and fix (reference + your code side by side)
  Read the reference implementation.
  Find every place yours differs.
  Understand WHY the reference made each choice.
  Fix yours, but keep your variable names and structure.
```

This is slower than copying, but it is the only way to own the knowledge.
Interviewers can tell the difference immediately.

### How to Know You Actually Understand Something

You understand a component if you can:
1. Explain it out loud without looking at code
2. Re-implement it from scratch after a 1-week break
3. Explain what would break if you removed it

If you cannot do all three, you do not own it yet.

---

## Project Structure

```
diffusion-policy-reimplementation/
├── diffusion_policy/                ← main package
│   ├── model/
│   │   ├── diffusion/
│   │   │   ├── scheduler.py         ← Stage 1A
│   │   │   ├── unet1d.py            ← Stage 1B–D
│   │   │   └── ema.py               ← Stage 6
│   │   └── vision/
│   │       └── encoder.py           ← Stage 4
│   ├── dataset/
│   │   ├── replay_buffer.py         ← Stage 2
│   │   ├── sampler.py               ← Stage 2
│   │   └── normalizer.py            ← Stage 2
│   └── policy/
│       ├── lowdim.py                ← Stage 3
│       └── image.py                 ← Stage 5
├── training/
│   └── workspace.py                 ← Stage 6
├── eval/
│   └── runner.py                    ← Stage 7
├── experiments/                     ← run these to validate each stage
│   ├── 01_ddpm_toy.py
│   ├── 02_dataset_test.py
│   └── 03_lowdim_train.py
├── train.py                         ← full pipeline entry point
├── eval.py
└── test_integration.py              ← run before full training (Stage 8)
```

The stage numbers live in this guide, not in directory names. Anyone reading the
repo sees a normal ML project structure; anyone following this guide knows exactly
which file maps to which stage.

---

## Stage 1 — Core Diffusion Mechanics

### What to Read First (before writing any code)

- DDPM paper (Ho et al. 2020): read sections 1, 2, 3, and 4 (skip appendix)
- Focus on: the forward process q(x_t | x_0), the reverse process p_θ(x_{t-1} | x_t),
  and the simplified training objective (equation 14 in the paper)

**The math you must internalize**:

```
Forward process (adding noise):
  x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε,   ε ~ N(0, I)

  Where:
  β_t  = noise schedule (increases from ~0.0001 to ~0.02)
  α_t  = 1 - β_t
  ᾱ_t  = ∏_{s=1}^{t} α_s   (cumulative product)

Training objective (simplified):
  L = E_{x_0, ε, t} [ ||ε - ε_θ(x_t, t)||² ]
  i.e., predict the noise that was added, not the clean signal

Reverse process (removing noise, one step):
  x_{t-1} = 1/√α_t · (x_t - β_t/√(1-ᾱ_t) · ε_θ(x_t, t)) + σ_t · z
  z ~ N(0, I)
```

Draw ᾱ_t as a function of t on paper. Understand that at t=0, x_t ≈ x_0
(clean), and at t=T, x_t ≈ pure noise.

---

### 1A. Write the Noise Scheduler

File: `diffusion_policy/model/diffusion/scheduler.py`

Write this yourself before looking at any library code:

```python
import torch
import numpy as np

class DDPMScheduler:
    def __init__(self, num_train_timesteps=100, beta_start=0.0001, beta_end=0.02,
                 beta_schedule='squaredcos_cap_v2'):
        self.num_train_timesteps = num_train_timesteps

        # TODO: implement beta schedule
        # 'linear':              torch.linspace(beta_start, beta_end, T)
        # 'squaredcos_cap_v2':  see the paper — cosine schedule is more stable
        self.betas = ...

        # TODO: compute alphas and cumulative alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Useful precomputed quantities for the forward and reverse formulas
        self.sqrt_alphas_cumprod = ...
        self.sqrt_one_minus_alphas_cumprod = ...

    def add_noise(self, x0, noise, timesteps):
        """Forward process: compute x_t from x_0 and noise."""
        # x0:        [B, H, D]
        # noise:     [B, H, D]  (same shape)
        # timesteps: [B]        (integer indices)
        # Returns:   x_t [B, H, D]
        # TODO: use sqrt_alphas_cumprod and sqrt_one_minus_alphas_cumprod
        # Hint: index them by timesteps, then reshape for broadcasting
        ...

    def step(self, eps_pred, t, x_t):
        """Reverse process: compute x_{t-1} from eps_pred and x_t."""
        # This is the DDPM sampling step (one denoising step)
        # See equation 11 in Ho et al. 2020
        # TODO: compute x_{t-1}
        # Return an object with attribute .prev_sample (to match diffusers API)
        ...

    def set_timesteps(self, num_inference_steps):
        """Set timesteps for inference (can be fewer than training timesteps)."""
        # For DDPM: just use all T steps in reverse
        self.timesteps = torch.arange(self.num_train_timesteps - 1, -1, -1)
```

**Validate**:

```python
# Test: add noise then check that variance matches theory
scheduler = DDPMScheduler(num_train_timesteps=100)
x0 = torch.zeros(1, 16, 2)   # clean zeros
noise = torch.ones(1, 16, 2)
t = torch.tensor([99])
xt = scheduler.add_noise(x0, noise, t)
# At t=99, xt should be ≈ noise (signal ≈ 0, noise ≈ 1)
# At t=0,  xt should be ≈ x0   (signal ≈ 1, noise ≈ 0)
print(xt)  # should be close to 1.0 everywhere
```

---

### 1B. Write the Sinusoidal Embedding

File: `diffusion_policy/model/diffusion/unet1d.py` (top of file)

```python
import math
import torch
import torch.nn as nn

class SinusoidalPosEmb(nn.Module):
    """
    Encode a scalar timestep t into a vector of dimension `dim`.
    Standard in transformers and diffusion models.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: [B]  (integer or float timestep indices)
        # Returns: [B, dim]
        # TODO: implement
        # Formula: emb[i] = sin(t / 10000^(2i/dim)) for even i
        #                    cos(t / 10000^(2i/dim)) for odd i
        # Hint: compute half_dim = dim // 2
        #       freqs = exp(-log(10000) * arange(half_dim) / half_dim)
        #       emb = t.unsqueeze(1) * freqs.unsqueeze(0)
        #       return cat([sin(emb), cos(emb)], dim=-1)
        ...
```

**Why this exists**: the denoiser needs to know which noise level it is operating
at. A learned embedding would work but sinusoidal is a good prior — it gives the
network a smooth, continuous signal.

---

### 1C. Write the 1D Residual Block with FiLM Conditioning

```python
class ConditionalResidualBlock1D(nn.Module):
    """
    1D conv residual block where the global condition modulates activations
    via FiLM (Feature-wise Linear Modulation): scale + shift.
    """
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3):
        super().__init__()
        # TODO: build these components:
        # - conv1: Conv1d(in_channels, out_channels, kernel_size, padding=same)
        # - conv2: Conv1d(out_channels, out_channels, kernel_size, padding=same)
        # - cond_proj: Linear(cond_dim, 2 * out_channels)  ← produces scale + shift
        # - residual_conv: Conv1d(in_channels, out_channels, 1) if in!=out else Identity
        # - GroupNorm on both conv outputs
        ...

    def forward(self, x, cond):
        # x:    [B, C, L]   (channels first for Conv1d)
        # cond: [B, cond_dim]
        # 1. Project cond → scale, shift   [B, out_channels] each
        # 2. x = norm(conv1(x))
        # 3. x = x * (1 + scale) + shift   ← FiLM modulation
        # 4. x = norm(conv2(x))
        # 5. x = x + residual_conv(input)  ← skip connection
        ...
```

**Why FiLM instead of concatenation**: FiLM (scale + shift) multiplicatively
gates the features based on the condition. This is more expressive than additive
concat — the condition can suppress or amplify specific feature channels.

---

### 1D. Write the 1D U-Net

```python
class ConditionalUnet1D(nn.Module):
    """
    1D U-Net that denoises an action sequence conditioned on:
    - diffusion timestep t (via sinusoidal embedding)
    - global observation embedding (via FiLM in each block)
    """
    def __init__(self, input_dim, global_cond_dim, down_dims=[256, 512, 1024]):
        super().__init__()
        # Architecture:
        #   input_proj: Linear(input_dim, down_dims[0])
        #   time_emb: SinusoidalPosEmb(dim) → Linear → Mish → Linear
        #   cond_dim = time_emb_dim + global_cond_dim
        #   down: list of ConditionalResidualBlock1D pairs + Downsample1d
        #   mid: ConditionalResidualBlock1D
        #   up:  list of ConditionalResidualBlock1D pairs + Upsample1d + skip concat
        #   output_proj: Conv1d(down_dims[0], input_dim, 1)

    def forward(self, sample, timestep, global_cond=None):
        # sample:      [B, H, action_dim]  — noisy action sequence
        # timestep:    [B]                 — diffusion step index
        # global_cond: [B, obs_dim]        — encoded observation
        # Returns:     [B, H, action_dim]  — predicted noise

        # Step 1: transpose to channels-first: [B, action_dim, H]
        # Step 2: compute time embedding
        # Step 3: concatenate time_emb and global_cond → cond
        # Step 4: run through down blocks, saving skip connections
        # Step 5: run through mid block
        # Step 6: run through up blocks, concatenating skip connections
        # Step 7: project back to action_dim
        # Step 8: transpose back to [B, H, action_dim]
        ...
```

---

### 1E. Validate on a Toy Dataset

File: `experiments/01_ddpm_toy.py`

```python
"""
Train a 1D diffusion model on a 2D bimodal distribution.
This validates stages 1A-1D without any robot data.
"""
import torch
import torch.nn as nn
from noise_scheduler import DDPMScheduler
from unet1d import ConditionalUnet1D

# Toy dataset: 2D points drawn from a mixture of two Gaussians
# (analogous to: action = [x, y], no temporal structure yet)
def sample_bimodal(n):
    mode = torch.randint(0, 2, (n,))
    means = torch.tensor([[-2.0, 0.0], [2.0, 0.0]])
    x = means[mode] + torch.randn(n, 2) * 0.3
    return x   # [N, 2]

# Treat each 2D point as a sequence of length 1: [B, 1, 2]
scheduler = DDPMScheduler(num_train_timesteps=100)
model = ConditionalUnet1D(input_dim=2, global_cond_dim=0, down_dims=[64, 128])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(5000):
    x0 = sample_bimodal(256).unsqueeze(1)      # [256, 1, 2]
    noise = torch.randn_like(x0)
    t = torch.randint(0, 100, (256,))
    xt = scheduler.add_noise(x0, noise, t)
    eps_pred = model(xt, t)
    loss = nn.functional.mse_loss(eps_pred, noise)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if step % 500 == 0:
        print(f"step {step}: loss={loss.item():.4f}")

# Generate samples and plot — should show two clusters at x≈-2 and x≈+2
scheduler.set_timesteps(100)
x = torch.randn(1000, 1, 2)
for t in scheduler.timesteps:
    eps_pred = model(x, t.expand(1000))
    x = scheduler.step(eps_pred, t, x).prev_sample

import matplotlib.pyplot as plt
pts = x.squeeze(1).detach().numpy()
plt.scatter(pts[:, 0], pts[:, 1], alpha=0.3, s=5)
plt.title("Samples from DDPM (should show two clusters)")
plt.savefig("toy_samples.png")
```

**You have completed Stage 1 when**: the scatter plot shows two distinct clusters.
If you see one blob in the middle, your denoiser is not learning multimodality.

---

## Stage 2 — Dataset + Normalizer

### What to Read First

Open the Zarr docs briefly, then focus on understanding the data format:

```python
import zarr
z = zarr.open('pusht_cchi_v7_replay.zarr', 'r')
print(z.tree())
# data/
#   img       (25650, 96, 96, 3)   uint8
#   action    (25650, 2)            float32
#   state     (25650, 5)            float32
# meta/
#   episode_ends (206,)             int64
```

The key insight: `episode_ends[i]` is the cumulative step count at the end of
episode i. So episode 0 goes from step 0 to step `episode_ends[0]-1`.

### Write the SequenceSampler

This is harder than it looks. Work through it carefully.

```python
class SequenceSampler:
    """
    Given a list of episode lengths, produces indices for all valid
    windows of length `sequence_length`, with padding at boundaries.
    """
    def __init__(self, episode_ends, sequence_length, pad_before=0, pad_after=0):
        # Build a list of (episode_idx, start_step_within_episode) for all
        # valid windows. A window is valid if it starts at a step that is
        # at least -pad_before from the episode start and ends at most
        # pad_after beyond the episode end.
        # This allows windows that include padded frames at boundaries.
        self.indices = []
        # TODO: for each episode, compute its valid start positions
        # and append (episode_id, start) to self.indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep_id, start = self.indices[idx]
        # TODO: gather sequence_length steps starting at `start`
        # For steps before episode start: repeat the first frame
        # For steps after episode end:   repeat the last frame
        # Return: dict of {key: [sequence_length, ...] arrays}
        ...
```

**Validate**: print the first 5 sampled windows and verify that:
- The shapes are `[H, ...]` for each key
- Frames at episode boundaries are correctly padded (first/last frame repeated)
- No steps from adjacent episodes bleed into a window

### Write the LinearNormalizer

```python
class LinearNormalizer:
    """
    Fit stats from data, normalize to [-1, 1] (or standardize to zero mean).
    Must handle nested dicts: normalizer['obs'].normalize(x)
    """
    def fit(self, data, mode='limits'):
        # data: tensor [N, D] or dict of tensors
        # mode='limits':  normalize to [-1, 1] using min/max
        # mode='gaussian': normalize to N(0,1) using mean/std
        if mode == 'limits':
            self.min = data.min(dim=0).values
            self.max = data.max(dim=0).values
        ...

    def normalize(self, x):
        # x: [..., D]
        # Returns: [..., D] in [-1, 1]
        ...

    def unnormalize(self, x):
        # Inverse of normalize
        ...
```

---

## Stage 3 — Lowdim Policy

### Before You Write Any Policy Code

On paper, trace through one training step manually:
1. Draw `batch['obs']` shape: `[B=64, To=2, obs_dim=5]`
2. Draw `batch['action']` shape: `[B=64, H=16, 2]`
3. Draw what `global_cond` looks like after flattening obs
4. Draw what `xt` looks like after adding noise
5. Draw what `eps_pred` looks like coming out of the U-Net

Only start coding after you can draw this data flow without looking at notes.

### Implement the Policy

```python
class DiffusionUnetLowdimPolicy(nn.Module):
    def __init__(self, model, noise_scheduler, horizon, obs_dim, action_dim,
                 n_obs_steps, n_action_steps, num_inference_steps=None):
        super().__init__()
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.num_inference_steps = num_inference_steps or noise_scheduler.num_train_timesteps
        self.normalizer = LinearNormalizer()

    def set_normalizer(self, normalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # TODO: implement step by step
        # 1. Extract obs, action from batch
        # 2. Normalize both
        # 3. Flatten obs → global_cond: [B, n_obs_steps * obs_dim]
        # 4. Sample t ~ Uniform[0, T)
        # 5. Sample ε ~ N(0, I)
        # 6. Compute noisy action: xt = scheduler.add_noise(action, ε, t)
        # 7. Predict noise: eps_pred = model(xt, t, global_cond)
        # 8. Return MSE(eps_pred, ε)
        ...

    @torch.no_grad()
    def predict_action(self, obs_dict):
        # TODO: implement inference loop
        # 1. Normalize obs
        # 2. Flatten obs → global_cond
        # 3. x = randn([B, horizon, action_dim])
        # 4. scheduler.set_timesteps(num_inference_steps)
        # 5. for t in scheduler.timesteps:
        #       eps_pred = model(x, t.expand(B), global_cond)
        #       x = scheduler.step(eps_pred, t, x).prev_sample
        # 6. unnormalize x
        # 7. return x[:, :n_action_steps]
        ...
```

### Write a Standalone Training Script

```python
# experiments/03_lowdim_train.py
# Train the lowdim policy on PushT state data
# Run this and verify loss goes below 0.1 in ~1000 steps

import torch
from torch.utils.data import DataLoader
# ... your imports

dataset = PushTDataset(zarr_path='...', horizon=16, n_obs_steps=2, n_action_steps=8)
normalizer = dataset.get_normalizer()
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

model = ConditionalUnet1D(input_dim=2, global_cond_dim=2*5, down_dims=[256, 512, 1024])
scheduler = DDPMScheduler(num_train_timesteps=100)
policy = DiffusionUnetLowdimPolicy(model=model, noise_scheduler=scheduler,
                                    horizon=16, obs_dim=5, action_dim=2, ...)
policy.set_normalizer(normalizer)

optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4, weight_decay=1e-6)

for epoch in range(100):
    for batch in dataloader:
        loss = policy.compute_loss(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"epoch {epoch}: loss={loss.item():.4f}")
```

---

## Stage 4 — Vision Encoder

### The GroupNorm Requirement (Critical)

Before writing anything, understand this completely:

```
EMA maintains a shadow copy of model weights that is a running average.
BatchNorm2d has running_mean and running_var buffers that are updated
during the forward pass (not during optimizer.step).
EMA does NOT update these buffers — it only copies the learned parameters.
Result: EMA model's BN statistics are stale → poor eval performance.
Fix: replace all BatchNorm2d with GroupNorm(num_groups, num_channels).
GroupNorm has no running statistics — it normalizes per group per sample.
So it is fully compatible with weight-space EMA.
```

### How to Build the Vision Encoder

```python
import torchvision.models as models

def replace_batchnorm_with_groupnorm(module, num_groups=32):
    """Recursively replace all BatchNorm2d with GroupNorm."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            # GroupNorm(num_groups, num_channels)
            # num_groups must divide num_channels evenly
            groups = min(num_groups, num_channels)
            setattr(module, name, nn.GroupNorm(groups, num_channels))
        else:
            replace_batchnorm_with_groupnorm(child, num_groups)

class MultiImageObsEncoder(nn.Module):
    def __init__(self, image_shape=(3, 96, 96), feature_dim=512,
                 lowdim_dim=2, n_obs_steps=2):
        super().__init__()
        self.n_obs_steps = n_obs_steps

        # Load ResNet-18 backbone, remove the final FC layer
        backbone = models.resnet18(pretrained=False)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # → [B, 512, 1, 1]
        replace_batchnorm_with_groupnorm(self.backbone)

        # TODO: compute output_dim
        self.lowdim_dim = lowdim_dim
        # output: n_obs_steps * (512 + lowdim_dim)
        self.output_dim = n_obs_steps * (512 + lowdim_dim)

    def forward(self, obs_dict):
        # obs_dict['image']:     [B, To, 3, H, W]
        # obs_dict['agent_pos']: [B, To, 2]
        B, To = obs_dict['image'].shape[:2]

        # Encode images: merge batch and time dims
        imgs = obs_dict['image'].reshape(B * To, 3, 96, 96)
        feats = self.backbone(imgs)           # [B*To, 512, 1, 1]
        feats = feats.flatten(1)              # [B*To, 512]
        feats = feats.reshape(B, To, 512)     # [B, To, 512]

        # Concatenate low-dim obs
        lowdim = obs_dict['agent_pos']        # [B, To, 2]
        combined = torch.cat([feats, lowdim], dim=-1)  # [B, To, 514]

        # Flatten time dimension
        return combined.reshape(B, -1)        # [B, To * 514]
```

**Validate**:

```python
encoder = MultiImageObsEncoder()
dummy = {
    'image': torch.randn(4, 2, 3, 96, 96),
    'agent_pos': torch.randn(4, 2, 2)
}
out = encoder(dummy)
print(out.shape)   # should be [4, 1028]  (2 * (512 + 2))
```

---

## Stage 5 — Image Policy

At this point you have all the pieces. The image policy is just:

```
MultiImageObsEncoder → global_cond → ConditionalUnet1D
```

```python
class DiffusionUnetHybridImagePolicy(nn.Module):
    def __init__(self, obs_encoder, model, noise_scheduler, horizon,
                 action_dim, n_obs_steps, n_action_steps):
        super().__init__()
        self.obs_encoder = obs_encoder
        self.model = model                    # ConditionalUnet1D
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        # ... store other params

    def compute_loss(self, batch):
        # 1. Normalize image + agent_pos + action
        # 2. Encode obs → global_cond via self.obs_encoder
        # 3. Identical to lowdim from here: sample t, add noise, predict, MSE
        ...

    def predict_action(self, obs_dict):
        # 1. Normalize obs
        # 2. Encode: global_cond = self.obs_encoder(obs_dict)
        # 3. Identical to lowdim: randn → denoising loop → unnormalize → slice
        ...
```

The key discipline here: notice that `compute_loss` and `predict_action` differ
from the lowdim version only in the obs encoding step. Everything else is the
same diffusion math.

---

## Stage 6 — EMA + Full Training Loop

### EMA (Exponential Moving Average)

```python
class EMAModel:
    """
    Maintains a shadow copy of model parameters as a running average.
    Used at inference time for more stable predictions.
    """
    def __init__(self, parameters, power=0.75, max_value=0.9999):
        # power and max_value control how quickly the decay factor ramps up
        # early in training, decay is low (new weights matter more)
        # later in training, decay is high (more stable shadow weights)
        self.shadow_params = [p.clone().detach() for p in parameters]
        self.power = power
        self.max_value = max_value
        self.step_count = 0

    def get_decay(self):
        # Decay ramps from 0 to max_value as step_count grows
        # Formula: 1 - (1 + step / gamma)^(-power)
        # gamma is implicitly baked in via power
        step = self.step_count
        decay = 1.0 - (1 + step) ** (-self.power)
        return min(decay, self.max_value)

    def step(self, parameters):
        decay = self.get_decay()
        self.step_count += 1
        for shadow, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                shadow.data = decay * shadow.data + (1 - decay) * param.data

    def copy_to(self, parameters):
        """Copy shadow weights into model parameters (for eval)."""
        for shadow, param in zip(self.shadow_params, parameters):
            param.data.copy_(shadow.data)
```

### Full Training Loop

```python
# training/workspace.py

def train(cfg):
    # 1. Build dataset + dataloaders
    dataset = PushTImageDataset(...)
    normalizer = dataset.get_normalizer()
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)

    # 2. Build policy
    obs_encoder = MultiImageObsEncoder(...)
    model = ConditionalUnet1D(input_dim=2, global_cond_dim=obs_encoder.output_dim, ...)
    scheduler = DDPMScheduler(num_train_timesteps=100)
    policy = DiffusionUnetHybridImagePolicy(obs_encoder, model, scheduler, ...)
    policy.set_normalizer(normalizer)
    policy = policy.cuda()

    # 3. EMA shadow copy
    ema = EMAModel(policy.parameters(), power=0.75, max_value=0.9999)
    ema_policy = copy.deepcopy(policy)

    # 4. Optimizer + LR schedule
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4, weight_decay=1e-6)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=500,
                                                    num_training_steps=total_steps)

    # 5. Training loop
    global_step = 0
    for epoch in range(3050):
        policy.train()
        for batch in train_loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            loss = policy.compute_loss(batch)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            ema.step(policy.parameters())
            global_step += 1

        # 6. Evaluate with EMA weights
        if epoch % 50 == 0:
            ema.copy_to(ema_policy.parameters())
            ema_policy.eval()
            scores = env_runner.run(ema_policy)
            print(f"epoch {epoch}: mean_score={scores['mean_score']:.3f}")
            # save checkpoint if best score
```

---

## Stage 7 — Evaluation

### Understand the Eval Loop

```python
@torch.no_grad()
def run_episode(env, policy, max_steps=300):
    obs = env.reset()
    obs_buffer = deque(maxlen=n_obs_steps)
    obs_buffer.append(obs)

    done = False
    total_reward = 0.0

    for _ in range(max_steps):
        # Pad obs buffer to n_obs_steps
        while len(obs_buffer) < n_obs_steps:
            obs_buffer.appendleft(obs_buffer[0])

        # Build obs dict with temporal stack
        obs_dict = {
            'image':     torch.stack([o['image'] for o in obs_buffer]),     # [To, 3, H, W]
            'agent_pos': torch.stack([o['agent_pos'] for o in obs_buffer]), # [To, 2]
        }
        obs_dict = {k: v.unsqueeze(0).cuda() for k, v in obs_dict.items()}  # add batch dim

        # Predict action sequence
        actions = policy.predict_action(obs_dict)   # [1, n_action_steps, 2]
        actions = actions.squeeze(0).cpu().numpy()  # [n_action_steps, 2]

        # Execute each predicted action
        for action in actions:
            obs, reward, done, info = env.step(action)
            obs_buffer.append(obs)
            total_reward = max(total_reward, reward)
            if done:
                break
        if done:
            break

    return total_reward
```

**Why execute multiple actions before re-planning**: this is called action chunking.
Replanning at every step causes jitter (the policy's future predictions are
inconsistent across consecutive timesteps). Executing 8 steps between replans
produces smoother motion.

---

## How to Use the Reference Code

The reference is at `Imitation-Learning/diffusion_policy-main/`.
After you write your own version of each component, compare side by side:

```bash
# Compare your noise scheduler to the reference
diff diffusion_policy/model/diffusion/scheduler.py \
  ../diffusion_policy-main/diffusion_policy/model/diffusion/scheduling_ddpm.py

# Compare your U-Net
diff diffusion_policy/model/diffusion/unet1d.py \
  ../diffusion_policy-main/diffusion_policy/model/diffusion/conditional_unet1d.py
```

When you find differences, ask: "Is my way also correct? Or is their way better?
Why?" This is more valuable than just fixing to match.

---

## What Interviewers at AI/Robotics Companies Will Ask

After you complete this project, you should be able to answer all of these:

**Diffusion fundamentals**
- Why does predicting noise (ε) work better than predicting x_0 directly?
- What is the difference between DDPM and DDIM? When would you use each?
- What is the β schedule and why does the cosine schedule outperform linear?
- What happens if you run too few denoising steps at inference?

**Imitation learning context**
- Why does a Gaussian policy fail on multimodal demonstrations?
- What is action chunking and why does it reduce error compounding?
- What is the role of the observation horizon (To)? Why use To=2 for PushT?
- How does inpainting conditioning differ from cross-attention conditioning?

**Engineering decisions**
- Why replace BatchNorm with GroupNorm when using EMA?
- How does FiLM conditioning work and why use it over naive concatenation?
- What is the Zarr format and why is it used for demonstration data?
- How would you scale this to 100 hours of robot demonstration data?

**Your implementation specifically**
- Walk me through your code for `compute_loss`. What does each line do?
- What was the hardest bug you hit? How did you debug it?
- What would you change if you had to run this on a real robot arm?

Prepare answers to all of these before any interview.

---

## Stage 8 — Integration (wire all stages into one runnable system)

After completing stages 1–7 in isolation, this stage connects them into a single
end-to-end trainable and evaluatable system. **This is what makes the project
actually run.**

### 8A. Canonical Import Map

Every file should import from your own stages only (except PushTImageEnv):

```python
# diffusion_policy/policy/lowdim.py
from diffusion_policy.model.diffusion.scheduler import DDPMScheduler
from diffusion_policy.model.diffusion.unet1d import ConditionalUnet1D
from diffusion_policy.dataset.normalizer import LinearNormalizer

# diffusion_policy/policy/image.py
from diffusion_policy.model.diffusion.scheduler import DDPMScheduler
from diffusion_policy.model.diffusion.unet1d import ConditionalUnet1D
from diffusion_policy.dataset.normalizer import LinearNormalizer
from diffusion_policy.model.vision.encoder import MultiImageObsEncoder

# training/workspace.py
from diffusion_policy.dataset.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.sampler import SequenceSampler
from diffusion_policy.dataset.normalizer import LinearNormalizer
from diffusion_policy.policy.image import DiffusionUnetHybridImagePolicy
from diffusion_policy.model.diffusion.ema import EMAModel

# eval/runner.py
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv  # reference only
from diffusion_policy.policy.image import DiffusionUnetHybridImagePolicy
```

### 8B. Single Entry Point

Create `train.py` at the repo root as the single command to run everything:

```python
# train.py  — run this to train the full image policy on PushT
import torch
import copy
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from diffusion_policy.dataset.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.sampler import SequenceSampler
from diffusion_policy.dataset.normalizer import LinearNormalizer
from diffusion_policy.model.vision.encoder import MultiImageObsEncoder
from diffusion_policy.model.diffusion.unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.scheduler import DDPMScheduler
from diffusion_policy.policy.image import DiffusionUnetHybridImagePolicy
from diffusion_policy.model.diffusion.ema import EMAModel
from training.workspace import train   # your train() function

# Config — change these to match your setup
DATA_PATH  = 'data/pusht_cchi_v7_replay.zarr'
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
NUM_EPOCHS = 3050
HORIZON    = 16
N_OBS_STEPS    = 2
N_ACTION_STEPS = 8

if __name__ == '__main__':
    train(
        zarr_path=DATA_PATH,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        horizon=HORIZON,
        n_obs_steps=N_OBS_STEPS,
        n_action_steps=N_ACTION_STEPS,
    )
```

### 8C. Integration Test (run before full training)

Before committing to a 3000-epoch run, verify the full pipeline works with a
tiny smoke test:

```python
# test_integration.py — run this first, takes < 30 seconds
import torch
from diffusion_policy.model.diffusion.scheduler import DDPMScheduler
from diffusion_policy.model.diffusion.unet1d import ConditionalUnet1D
from diffusion_policy.dataset.normalizer import LinearNormalizer
from diffusion_policy.model.vision.encoder import MultiImageObsEncoder
from diffusion_policy.policy.image import DiffusionUnetHybridImagePolicy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Build all components
obs_encoder = MultiImageObsEncoder(n_obs_steps=2).to(device)
model = ConditionalUnet1D(
    input_dim=2,
    global_cond_dim=obs_encoder.output_dim,
    down_dims=[64, 128, 256],   # small dims for fast test
).to(device)
scheduler = DDPMScheduler(num_train_timesteps=100)
policy = DiffusionUnetHybridImagePolicy(
    obs_encoder=obs_encoder,
    model=model,
    noise_scheduler=scheduler,
    horizon=16,
    action_dim=2,
    n_obs_steps=2,
    n_action_steps=8,
).to(device)

# Fake batch
batch = {
    'image':     torch.randn(4, 2, 3, 96, 96).to(device),
    'agent_pos': torch.randn(4, 2, 2).to(device),
    'action':    torch.randn(4, 16, 2).to(device),
}

# Fit a dummy normalizer
normalizer = LinearNormalizer()
normalizer.fit({'image': batch['image'].reshape(-1, 3, 96, 96),
                'agent_pos': batch['agent_pos'].reshape(-1, 2),
                'action': batch['action'].reshape(-1, 2)})
policy.set_normalizer(normalizer)

# Test training step
loss = policy.compute_loss(batch)
loss.backward()
print(f"compute_loss OK: {loss.item():.4f}")

# Test inference
policy.eval()
obs_dict = {'image': batch['image'][:1], 'agent_pos': batch['agent_pos'][:1]}
actions = policy.predict_action(obs_dict)
print(f"predict_action OK: shape={actions.shape}")   # should be [1, 8, 2]

print("All integration tests passed.")
```

Run `python test_integration.py` before starting full training. If this passes,
the full training run will work.

### 8D. End-to-End Checklist

```
□ conda env created and activated
□ pip install -e . (package installed in editable mode)
□ pip install -e ../diffusion_policy-main (for PushTImageEnv only)
□ data/pusht_cchi_v7_replay.zarr downloaded and accessible
□ python test_integration.py → "All integration tests passed"
□ python train.py → loss decreasing in first 100 steps
□ Checkpoint saved at epoch 50
□ python eval.py --checkpoint <path> → mean_score printed
□ mean_score > 0.70 on 50 test episodes
```

---

## Recommended Study Schedule

| Week | Focus | Deliverable |
|---|---|---|
| 0 | Setup (Stage 0) | Conda env, data downloaded, package installed |
| 1 | DDPM paper + Stage 1 | `toy_samples.png` showing bimodal distribution |
| 2 | Stage 2 (dataset) | Unit tests for SequenceSampler and Normalizer |
| 3 | Stage 3 (lowdim) | Loss curve converging, lowdim eval >0.5 |
| 4 | Stage 4 (vision) | Shape tests passing, GroupNorm validated |
| 5 | Stage 5+6 (image policy + training) | First image policy training run |
| 6 | Stage 7+8 (eval + integration) | `test_integration.py` passes, mean_score >0.7 |
| 7 | Polish + write-up | README with results, WandB screenshots |

After week 7 you have a concrete, verifiable portfolio project.

---

## How to Write Up This Project for Your Resume/Portfolio

1. **GitHub repo**: clean code, clear README, WandB training curves, video of the
   robot succeeding at PushT
2. **One-line summary**: "Reimplemented Diffusion Policy (Chi et al. 2023) from
   scratch in PyTorch, achieving X% mean success rate on the PushT benchmark"
3. **Blog post** (optional but high value): write up what you learned — especially
   the non-obvious decisions (GroupNorm, action chunking, inpainting masking)
4. **Extend it** (to stand out): after the baseline works, try one of:
   - Replace U-Net denoiser with a Transformer
   - Implement DDIM for fast inference and measure the speed/quality tradeoff
   - Apply it to a different task (robomimic, or a sim task of your choice)
   - Implement classifier-free guidance for stronger conditioning
