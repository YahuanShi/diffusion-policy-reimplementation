"""
Image-conditioned Diffusion Policy — the core implementation for image observations.

Training (always DDPM forward process):
  1. Image observations → ResNet encoding → flattened obs_features
  2. Normalize action sequence → sample random t → add noise via DDPM forward process
  3. U-Net predicts noise ε_θ(x_t, t, obs) → MSE loss against true noise

Inference (DDIM reverse process by default):
  1. Encode current observations → obs_features (global conditioning)
  2. Start from pure noise x_T ~ N(0,I)
  3. Run DDIM reverse: 16 deterministic steps → normalized action sequence
     (DDPMScheduler also works; DDIM is ~6× faster with same quality)
  4. Unnormalize → extract action[To-1 : To-1+n_action_steps] as the executed chunk

Scheduler choice is made at load time (inference.py / eval.py), not here.
This class accepts any scheduler implementing set_timesteps() / step().

Key design — obs as global conditioning:
  Obs features are injected via FiLM into every U-Net ResBlock, not concatenated
  to the action sequence. The U-Net denoises actions only; obs acts as a global signal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.diffusion.scheduler import DDPMScheduler
from diffusion_policy.model.diffusion.unet1d import ConditionalUnet1D
from diffusion_policy.model.vision.encoder import MultiImageObsEncoder
from diffusion_policy.dataset.normalizer import LinearNormalizer


class DiffusionUnetImagePolicy(nn.Module):
    def __init__(
        self,
        obs_encoder: MultiImageObsEncoder,
        noise_scheduler: DDPMScheduler,
        shape_meta: dict,
        horizon,
        n_obs_steps,
        n_action_steps,
        num_inference_steps=None,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
    ):
        super().__init__()
        action_dim = shape_meta["action"]["shape"][0]
        obs_feature_dim = obs_encoder.output_shape()[0]

        # global_cond_dim = obs_feature_dim × n_obs_steps
        # e.g. ResNet18 outputs 512-dim, n_obs_steps=2 → global_cond = 1024-dim
        model = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_feature_dim * n_obs_steps,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=list(down_dims),
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.action_dim = action_dim
        self.obs_feature_dim = obs_feature_dim
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.num_inference_steps = (
            num_inference_steps or noise_scheduler.num_train_timesteps
        )

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _encode_obs(self, obs_dict, n_steps):
        """
        Encode observations into a global conditioning vector.

        Input: each value in obs_dict has shape (B, T, ...)
        Steps:
          1. Slice first n_steps: (B, T, ...) → (B, n_steps, ...)
          2. Normalize low_dim data (images are NOT normalized here — ResNet handles that)
          3. Flatten time dim: (B, n_steps, ...) → (B*n_steps, ...)
          4. Pass through MultiImageObsEncoder → (B*n_steps, feat_dim)
          5. Reshape → (B, n_steps * feat_dim) as global_cond
        """
        B = None
        flat_obs = {}
        low_dim_keys = set(
            k
            for k, v in self.obs_encoder.shape_meta["obs"].items()
            if v.get("type") == "low_dim"
        )

        for key, val in obs_dict.items():
            if B is None:
                B = val.shape[0]
            sliced = val[:, :n_steps]
            if key in low_dim_keys and key in self.normalizer:
                sliced = self.normalizer[key].normalize(sliced)
            flat_obs[key] = sliced.reshape(-1, *sliced.shape[2:])

        features = self.obs_encoder(flat_obs)
        return features.reshape(B, -1)

    def conditional_sample(self, shape, global_cond):
        device = global_cond.device
        scheduler = self.noise_scheduler
        trajectory = torch.randn(size=shape, device=device)

        scheduler.set_timesteps(self.num_inference_steps)
        for t in scheduler.timesteps:
            model_output = self.model(trajectory, t, global_cond=global_cond)
            trajectory = scheduler.step(model_output, t, trajectory).prev_sample

        return trajectory

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Training loss computation.

        Steps:
          1. Normalize actions to [-1, 1]
          2. Encode observations into global_cond
          3. Sample random timestep and noise
          4. Add noise: x_t = sqrt(alpha_bar_t) * action + sqrt(1-alpha_bar_t) * eps
          5. U-Net predicts noise: eps_theta = model(x_t, t, global_cond)
          6. Loss: MSE(eps_theta, eps)
        """
        naction = self.normalizer["action"].normalize(batch["action"])
        B = naction.shape[0]

        obs_dict = {k: v for k, v in batch.items() if k != "action"}
        global_cond = self._encode_obs(obs_dict, self.n_obs_steps)

        noise = torch.randn_like(naction)
        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps, (B,), device=naction.device
        ).long()

        noisy_action = self.noise_scheduler.add_noise(naction, noise, timesteps)
        eps_pred = self.model(noisy_action, timesteps, global_cond=global_cond)

        loss = F.mse_loss(eps_pred, noise)
        return loss

    @torch.no_grad()
    def predict_action(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Inference: generate action sequence from noise.

        Steps:
          1. Encode observations
          2. DDPM reverse denoising: x_T → x_{T-1} → ... → x_0
          3. Unnormalize to get real actions
          4. Extract action[To-1 : To-1+n_action_steps]

        Timeline (horizon=16, To=2, n_action_steps=8):
          action_ts: [t-1, t, t+1, ..., t+14]
                      0    1   2         15
          obs covers positions 0..To-1 = [t-1, t]
          Execute from position To-1 = 1 (current time t):
          [t-1 | t  t+1 t+2 t+3 t+4 t+5 t+6 t+7 t+8 | t+9..t+14]
           obs   ^^^ start=To-1          end=To-1+8 ^^^
        """
        B = next(iter(obs_dict.values())).shape[0]
        To = self.n_obs_steps

        global_cond = self._encode_obs(obs_dict, To)

        shape = (B, self.horizon, self.action_dim)
        naction_pred = self.conditional_sample(shape, global_cond)

        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        return action
