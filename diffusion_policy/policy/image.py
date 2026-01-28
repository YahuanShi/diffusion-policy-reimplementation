import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.diffusion.scheduler import DDPMScheduler
from diffusion_policy.model.diffusion.unet1d import ConditionalUnet1D
from diffusion_policy.model.vision.encoder import MultiImageObsEncoder
from diffusion_policy.dataset.normalizer import LinearNormalizer


class DiffusionUnetImagePolicy(nn.Module):
    def __init__(self, obs_encoder: MultiImageObsEncoder,
                 noise_scheduler: DDPMScheduler,
                 shape_meta: dict,
                 horizon, n_obs_steps, n_action_steps,
                 num_inference_steps=None,
                 diffusion_step_embed_dim=256,
                 down_dims=(256, 512, 1024),
                 kernel_size=5,
                 n_groups=8,
                 cond_predict_scale=True):
        super().__init__()
        action_dim = shape_meta['action']['shape'][0]
        obs_feature_dim = obs_encoder.output_shape()[0]

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
        self.num_inference_steps = num_inference_steps or noise_scheduler.num_train_timesteps

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _encode_obs(self, obs_dict, n_steps):
        B = None
        flat_obs = {}
        for key, val in obs_dict.items():
            if B is None:
                B = val.shape[0]
            flat_obs[key] = val[:, :n_steps].reshape(-1, *val.shape[2:])

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

    def compute_loss(self, batch):
        naction = self.normalizer['action'].normalize(batch['action'])
        B = naction.shape[0]

        obs_dict = {k: v for k, v in batch.items() if k != 'action'}
        global_cond = self._encode_obs(obs_dict, self.n_obs_steps)

        noise = torch.randn_like(naction)
        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps, (B,)).long()

        noisy_action = self.noise_scheduler.add_noise(naction, noise, timesteps)
        eps_pred = self.model(noisy_action, timesteps, global_cond=global_cond)

        loss = F.mse_loss(eps_pred, noise)
        return loss

    @torch.no_grad()
    def predict_action(self, obs_dict):
        B = next(iter(obs_dict.values())).shape[0]
        To = self.n_obs_steps

        global_cond = self._encode_obs(obs_dict, To)

        shape = (B, self.horizon, self.action_dim)
        naction_pred = self.conditional_sample(shape, global_cond)

        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        start = To
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        return action
