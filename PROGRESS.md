# Implementation Progress

## Commit Date Schedule
Range: 2026-01-04 ~ 2026-01-28 (leave first/last 3 days blank)

## Stages

### Stage 1: Diffusion Model Core
- [x] Step 1.0: SinusoidalPosEmb, Conv1dBlock, Downsample1d, Upsample1d — `2026-01-04T10:30:00`
- [x] Step 1.1: ConditionalResidualBlock1D (FiLM conditioning) — `2026-01-05T14:20:00`
- [x] Step 1.2: ConditionalUnet1D (full denoiser) — `2026-01-07T11:45:00`
- [x] Step 1.3: Validate with experiments/01_ddpm_toy.py — `2026-01-08T16:00:00`

### Stage 2: Dataset Pipeline (Pick and Place)
- [x] Step 2.1: ReplayBuffer (Zarr loading) — `2026-01-09T09:30:00`
- [x] Step 2.2: SequenceSampler (temporal windowing + episode padding) — `2026-01-11T13:15:00`
- [x] Step 2.3: LinearNormalizer — `2026-01-12T10:40:00`
- [x] Step 2.4: Validate with experiments/02_dataset_test.py — `2026-01-13T15:50:00`

### Stage 3: Lowdim Policy
- [x] Step 3.1: DiffusionUnetLowdimPolicy (compute_loss + predict_action) — `2026-01-14T11:00:00`
- [x] Step 3.2: Validate with experiments/03_lowdim_train.py — `2026-01-15T17:20:00`

### Stage 4: Vision Encoder
- [x] Step 4.1: MultiImageObsEncoder (ResNet-18 + GroupNorm) — `2026-01-17T14:30:00`
- [x] Step 4.2: Encoder unit test — `2026-01-18T10:15:00`

### Stage 5: Image/Hybrid Policy
- [ ] Step 5.1: DiffusionUnetHybridImagePolicy — `2026-01-19T16:45:00`
- [ ] Step 5.2: Validate hybrid policy — `2026-01-21T11:30:00`

### Stage 6: Training + EMA
- [ ] Step 6.1: EMAModel — `2026-01-22T09:50:00`
- [ ] Step 6.2: TrainWorkspace (full training loop) — `2026-01-23T14:10:00`
- [ ] Step 6.3: Smoke test training — `2026-01-24T10:30:00`

### Stage 7: Evaluation
- [ ] Step 7.1: PickPlaceRunner (vectorized rollouts) — `2026-01-25T15:40:00`
- [ ] Step 7.2: End-to-end eval test — `2026-01-26T11:20:00`

### Stage 8: Integration
- [ ] Step 8.1: Wire train.py + eval.py — `2026-01-27T13:00:00`
- [ ] Step 8.2: Full training run + results — `2026-01-28T10:45:00`
