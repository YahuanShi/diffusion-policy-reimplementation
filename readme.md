# Diffusion Policy Reimplementation

A structured reimplementation and analysis of Diffusion Policy for robot learning, focusing on visuomotor policy learning via conditional action diffusion.

---

## 1. Objectives

### 1.1 Mechanism Understanding
Analyze how diffusion models are applied in action space to represent multimodal action distributions.  
Focus on:
- Conditional denoising process
- Observation-conditioned action generation
- Relationship between diffusion process and policy learning

---

### 1.2 Pipeline Reproduction
Reproduce the end-to-end training and evaluation pipeline, including:
- Dataset loading and preprocessing
- Visuomotor policy learning setup
- Action sequence prediction and sampling
- Environment interaction for evaluation

---

## 2. Diffusion Policy

Diffusion Policy is robot learning framework from [Chi et al., 2023 (RSS)](https://diffusion-policy.cs.columbia.edu/)  that applies denosing diffusion probabilistic models(DDPM) to imitation learning. Instead of derectly regressing actions, the policy learns to denoise random noise into a sequence of future actions conditioned on observations.

The key insight: representing the acrion distribution as a learned reserse diffusion process captures multimodal behavior (e.g., going left or right around an obstacle) far better thant MSE regression or Guassian policies.

---

### 2.1 Core Concepts

---

#### 2.1.1 The Diffusion Policy Idea

Standard BC maps observation directly to action (unimodal, MSE regression):

```
π(obs) → action
```

Diffusion Policy instead learn the reverse of noise-corruption process:

```
Froward: action -> noise action  (add Guassian noise over Tsteps)
Reverse: noise_action -> action  (learned denoiser conditioned on obs)
```
At inference: sample pure Guassian noise, run 100 denoising steps conditioned on the current observation -> profuces a clean action sequence.

**Why this works**: representing the action distribution as a reverse diffusion process naturally captures **multimodal behavior** (e.g., going left or right around an obstacle) that MSE regression or Guassion policies collapse to the mean.

---

### 2.1.2 Temporal Structure (Action Chuncking)
The policy reasons over a temporal window, not a single timesteo:

```
Timeline:
    [obs {t-1}, obs_t] -> PREDICT [a_t, a_{t-1}, ..., a_{t+7}]
     ↑ To=2 obs steps              ↑ Ta=8 action steps
     |<------------- Horizon H=16 ------------>|
```

- **Observation horizon** `To=2`: condition on last 2 image frames
- **Action horizon** `Ta=8`: predict 8 future actions at once
- **Execite** only the first `n_action_steps` action before re-planing

This receding-horizon control reduces compouding errors vs. single-step prediction.

---

### 2.1.3 Two Network Backbones

| Backbone | Class | Conditioning mechanism |
| --- | --- | --- |
| **1D U-Net** | `Conditioning mechanism` | Global: concat obsemb + step emb iva FiLM |
| **Transformer** | `TransformerForDiffusion` | Cross-attention over obs tokens |

### 2.1.4 Two Observation Modes
| Mode | Input | Vision encoder |
| --- | --- | --- |
| **Lowdim** | State vector (joint angles, pos, etc.) | None |
| **Image / Hybrid** | RGB image +low-dim state |ResNet-18/34 (GroupNorm) |

---

## 3. Project Architecture
```
diffusion_policy/
├── 
├──
│   ├── 