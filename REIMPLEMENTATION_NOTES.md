# Reimplementation Notes

Lessons learned from reimplementing Diffusion Policy from scratch.
Intended as a reference for future reimplementations (ACT, etc.).

---

## 1. Before Writing a Single Line of Code

**Read the paper and original code in parallel, not sequentially.**
The paper describes the ideal; the code contains the real design decisions. Key things that only appear in the code:
- GroupNorm instead of BatchNorm
- The specific EMA power law schedule
- clip_sample=True in the reverse process (critical for stability)
- `steps_per_inference` < `n_action_steps` (execute fewer than predicted)

**Identify the minimal reproducible architecture first.**
Before building the full pipeline, make sure you can:
1. Run a forward pass (training loss computes)
2. Run a reverse pass (inference produces an action)
3. Overfit on a single batch

Only then build data loading, logging, and evaluation around it.

---

## 2. Architecture — What Actually Matters

### Normalization

The normalizer is more important than it looks.

- Use **limits mode** (scale to [-1, 1]): required because DDPM `clip_sample=True` assumes actions live in [-1, 1].
- Normalize **both action and state** independently. The original paper only normalizes action; normalizing state prevents large state values from dominating concatenated features (inspiration: OpenPI).
- Fit the normalizer once on the full dataset before training, not per-batch.
- Performance: use **column access** `torch.tensor(hf_dataset[key])` not row-by-row iteration — orders of magnitude faster on large datasets.

### BatchNorm → GroupNorm

**Always replace BatchNorm with GroupNorm in robot learning.**

BatchNorm relies on per-batch statistics. Robot datasets use small batches (4–64) and batch statistics are unstable. GroupNorm has no running statistics, which is also why it's compatible with EMA — EMA copies buffers directly rather than smoothing them.

Rule of thumb: `num_groups = num_channels // 16`.

### EMA

EMA needs to handle **buffers**, not just parameters. A naive implementation using `model.parameters()` will miss running stats, positional embeddings, and any other registered buffers.

Correct pattern:
```python
for module, ema_module in zip(model.modules(), ema_model.modules()):
    for p, ep in zip(module.parameters(recurse=False), ema_module.parameters(recurse=False)):
        ep.mul_(decay).add_(p.data, alpha=1 - decay)
    for b, eb in zip(module.buffers(recurse=False), ema_module.buffers(recurse=False)):
        eb.copy_(b.data)  # buffers are hard-copied, not smoothed
```

Use EMA weights only for inference, not for computing the training loss.

### Conditioning Mechanism

Diffusion Policy uses **global FiLM conditioning**: obs features are injected into every U-Net ResBlock via scale/shift, not concatenated to the action sequence.

This means:
- The U-Net input is always `(B, H, action_dim)` — pure action sequence
- Observation only enters via FiLM
- This keeps the denoising problem clean: the network denoises actions, conditioned on obs

Alternative: **local conditioning** (concat obs to action at each position). More expressive but harder to train and not needed for fixed-camera setups.

### Data Augmentation

Use `RandomCenterCrop` instead of two separate transforms:
- Training: random crop (data augmentation)
- Eval/inference: center crop (deterministic)

Implemented as a single `nn.Module` that checks `self.training`. This ensures the behavior automatically switches when you call `policy.train()` / `policy.eval()`.

---

## 3. DDPM vs DDIM — The Most Confusing Part

**Training always uses DDPM** (forward process only: add noise, predict noise, MSE loss).  
**Inference can use either** — they share the same trained weights.

| | DDPM | DDIM |
|---|---|---|
| Type | Stochastic | Deterministic |
| Steps | 100 | 16 |
| Speed | Baseline | ~6× faster |
| Quality | Same | Same |
| When | Training forward / baseline inference | Deployment default |

The key bug to watch for: `set_timesteps(N)` must subsample **uniformly from T-1 to 0**, not just return `range(N-1, -1, -1)`.

Correct:
```python
self.timesteps = torch.linspace(T - 1, 0, N).long()  # e.g. [99, 93, 87, ..., 0] for N=16
```

Wrong:
```python
self.timesteps = torch.arange(N - 1, -1, -1)  # [15, 14, ..., 0] — ignores most of the schedule
```

DDIM also needs `alpha_prev` aligned to the subsampled schedule, not the full schedule.

---

## 4. Data Pipeline

### LeRobot Format

Modern standard. Uses parquet (metadata) + mp4 (video). Key concepts:
- `delta_timestamps`: relative time offsets for observation/action windows
- `hf_dataset`: HuggingFace dataset under the hood — use column access for speed
- Videos decoded per-frame on demand — `video_backend='pyav'` is faster than `torchvision`

### Timestamp Alignment (bug found and fixed)

A subtle but important issue for obs-conditioned policies. The first version had:

- **Training**: `delta_timestamps = [0, 1/fps]` → obs covers `[t, t+1]` (current + future)
- **Inference**: obs history = `[t-1, t]` (past + current)

These are misaligned. Fixed by anchoring both obs and action windows in the past:
- obs: `[-(To-1)/fps, ..., 0]`
- action: `[-(To-1)/fps, ..., (H-To)/fps]`, and execute `action[To-1 : To-1+Ta]` (starts at the current time t)

**Lesson**: the model never crashes on this — it just learns a slightly wrong mapping. Write the training window and the inference deque side by side on paper and check that index `To-1` means "now" in both. Any checkpoint trained before the fix must be retrained.

### Normalizer Fit

Fit on the full dataset split you train on. Do not fit on val/test. Do not re-fit at inference — load the normalizer state_dict from the checkpoint.

---

## 5. Inference Engineering

### Timing: `precise_wait`

Standard `time.sleep` has ~1ms jitter, which compounds to visible stuttering at 10–30 Hz.

Use a hybrid: sleep until ~1ms before the deadline, then busy-wait:
```python
def precise_wait(t_end):
    t_slack = 0.001
    t_sleep = t_end - time.time() - t_slack
    if t_sleep > 0:
        time.sleep(t_sleep)
    while time.time() < t_end:
        pass
```

### `steps_per_inference` vs `n_action_steps`

- `n_action_steps`: how many actions the model predicts per inference call (8)
- `steps_per_inference`: how many of those actions you actually execute (6)

Execute fewer than predicted so the policy can re-plan before the prediction horizon expires. This reduces compounding errors from imperfect action predictions. Original paper uses ~75% of predicted horizon.

### Latency Compensation Must Not Cross the Cycle Boundary

Skipping the action slots that elapsed during inference is the right idea, but execution must still **end** at the cycle boundary (`actions[exec_start:steps_per_inference]`), not run `steps_per_inference` actions starting from `exec_start`. Otherwise every cycle overruns by `exec_start` slots → the next cycle starts late → skips more → overruns more, until only one action per cycle is executed. Keep this logic in a pure function (`plan_execution`) so it can be unit-tested without a robot.

### Verify Conventions Against the Data, Not the Code

Units and sign/polarity conventions are where real-robot deployments fail silently. This project shipped with the gripper channel inverted at inference (code assumed 1 = closed, data uses 1 = open) — the model would have closed the gripper right after start and opened it during the carry. Checks that take minutes:
- Plot the gripper channel per episode: every pick-and-place demo starts and ends **open**, so the value at frame 0 *is* the "open" value.
- Look at one camera frame before and one during the grasp next to the recorded value.
- Compare action and state ranges with what the robot API returns (degrees vs radians).
- Compare the robot's home pose with the first frame of the demonstrations.

### Never Block the Control Loop on Sensors

`wait_for_frames()` on two cameras plus a blocking serial read easily exceed a 50 ms period, and a serial `readline()` returns the *oldest* buffered line — a stale reading. Read each sensor in its own thread, cache only the latest value with a timestamp, and let `get_obs()` copy the cache. Log `get_obs` time, action lateness and sensor age every cycle, and check the effective control rate in a dry run.

### Warm-up

Run 1–2 inference cycles before the robot moves to:
1. JIT-compile the model (first forward pass is slow)
2. Pre-load the first observation into the deque

### Observation Deque

Use `collections.deque(maxlen=n_obs_steps)` initialized with copies of the first observation. This gives a valid conditioning vector before enough history has accumulated.

---

## 6. Engineering Patterns That Scale

### Config Saving

Save `config.json` to `output_dir` at the start of every training run. This costs nothing and prevents the "what hyperparameters did I use for this checkpoint?" problem permanently.

Better still: store the same config **inside every checkpoint** and rebuild the model from it at load time (`diffusion_policy/policy/loader.py`). Hardcoding `horizon` / `down_dims` / `crop_shape` in eval and inference scripts fails silently — ResNet's adaptive pooling accepts any image size, so a missing center-crop is a distribution shift, not an error.

### Checkpoint Strategy

Train by **gradient steps** (`--max_steps`), not epochs: the step count is what the LR schedule and EMA warmup depend on, and it stays comparable when the dataset size changes. Save a step checkpoint every N steps and keep only the latest few (sort by step number, not lexicographically — `step10000` < `step5000` as strings).

What goes into a resumable checkpoint matters:
- `policy_state_dict` = **raw** training weights, `ema_state_dict` = EMA weights, plus optimizer state.
- Saving the EMA weights as `policy_state_dict` looks harmless for inference, but on resume they get loaded into the trained model: optimisation silently restarts from the smoothed weights, out of sync with the optimizer moments.
- Inference loads EMA from `ema_state_dict` when present; the final checkpoint has EMA already copied in.
- Replay the LR schedule (`global_step` × `scheduler.step()`) instead of loading its state — the saved state carries the old `T_max`, which is wrong if `--max_steps` changed.

### Validation Without a Simulator

Hold out a few **episodes** (not frames — neighbouring frames are near-duplicates, so a frame split leaks) and periodically score the EMA model: run the full reverse process and compare the executed chunk with the recorded actions. Fit the normalizer on training episodes only. This is a regression / checkpoint-selection signal, not a success rate: with multimodal demos a valid alternative trajectory still counts as error.

### Wandb Artifacts

Upload `policy_final.pt` as a wandb artifact at the end of training. This links the model weights to the exact run config and loss curve — essential when you have multiple experiments.

### Test Structure

For a policy reimplementation, three test levels are sufficient:

| Level | What to test | Why |
|---|---|---|
| Unit | Scheduler math, normalizer round-trip, EMA decay | Catches silent numerical bugs |
| Integration | Full forward + backward + predict_action | Catches shape/interface bugs |
| Smoke | `pytest -x` in CI | Catches regressions on push |

The scheduler unit tests are especially important — the DDPM/DDIM math is subtle and easy to get slightly wrong in ways that don't crash but produce bad results.

---

## 7. What to Reuse for ACT

ACT shares several components with Diffusion Policy that can be reused directly:

| Component | Reusable? | Notes |
|---|---|---|
| `MultiImageObsEncoder` | Yes | Same ResNet18 + GroupNorm pattern |
| `LinearNormalizer` | Yes | Same limits-mode normalization |
| `LeRobotImageDataset` | Yes | Same data format |
| `EMAModel` | Yes | Same buffer-aware implementation |
| `train.py` / `workspace.py` structure | Mostly | Replace policy class, keep training loop skeleton |
| `inference.py` timing logic | Yes | `precise_wait`, obs deque, steps_per_inference |
| GitHub Actions CI | Yes | Copy `.github/workflows/ci.yml` directly |
| Pre-commit config | Yes | Copy `.pre-commit-config.yaml` directly |

**Key differences in ACT:**
- Generator: CVAE encoder + transformer decoder (not U-Net)
- No diffusion process: direct action sequence regression from latent
- Conditioning: cross-attention to image tokens (not FiLM)
- KL loss term from CVAE in addition to action reconstruction loss
- At inference: encoder is discarded, only decoder runs (no noise)
- Typical horizon: 100 steps (longer than DP's 16)

**What to build first for ACT:**
1. CVAE encoder (ResNet/ViT image encoder → latent z)
2. Transformer decoder (z + obs features → action sequence)
3. Loss: reconstruction MSE + KL divergence weight β
4. Wrap in the same `compute_loss` / `predict_action` interface

---

## 8. Common Mistakes Checklist

Before claiming the reimplementation works, verify:

- [ ] `set_timesteps` uses `linspace(T-1, 0, N)`, not `arange(N)`
- [ ] EMA iterates over `module.buffers()`, not just `model.parameters()`
- [ ] Normalizer is fitted on training data only
- [ ] `policy.eval()` is called before inference (disables RandomCenterCrop, dropout, etc.)
- [ ] Images are **not** normalized with ImageNet stats (unless `imagenet_norm=True` + pretrained weights)
- [ ] `clip_sample=True` in scheduler step (critical for DDPM stability)
- [ ] `global_cond` dim = `obs_feature_dim * n_obs_steps`, not just `obs_feature_dim`
- [ ] Checkpoint saves `shape_meta` alongside weights (needed to reconstruct architecture at load time)
- [ ] `normalizer.state_dict()` is saved and loaded with the checkpoint
- [ ] Gripper / binary channels use the same polarity at inference as in the data (check frame 0 of every episode)
- [ ] Resumable checkpoints store raw weights in `policy_state_dict` and EMA separately; inference uses EMA
- [ ] Training config is stored in the checkpoint and used to rebuild the model at load time
- [ ] `ruff check` passes before every commit
