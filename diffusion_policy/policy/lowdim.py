import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.diffusion.scheduler import DDPMScheduler
from diffusion_policy.model.diffusion.unet1d import ConditionalUnet1D
from diffusion_policy.dataset.normalizer import LinearNormalizer


class DiffusionUnetLowdimPolicy(nn.Module):
    def __init__(self, model: ConditionalUnet1D, noise_scheduler: DDPMScheduler,
                 horizon, obs_dim, action_dim, n_obs_steps, n_action_steps,
                 num_inference_steps=None):
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

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def conditional_sample(self, shape, global_cond):
        scheduler = self.noise_scheduler
        trajectory = torch.randn(size=shape)

        scheduler.set_timesteps(self.num_inference_steps)
        for t in scheduler.timesteps:
            model_output = self.model(trajectory, t, global_cond=global_cond)
            trajectory = scheduler.step(model_output, t, trajectory).prev_sample

        return trajectory

    def compute_loss(self, batch):
        nobs = self.normalizer['obs'].normalize(batch['obs'])
        naction = self.normalizer['action'].normalize(batch['action'])
        B = naction.shape[0]

        global_cond = nobs[:, :self.n_obs_steps, :].reshape(B, -1)

        noise = torch.randn_like(naction)
        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps, (B,)).long()

        noisy_action = self.noise_scheduler.add_noise(naction, noise, timesteps)
        eps_pred = self.model(noisy_action, timesteps, global_cond=global_cond)

        loss = F.mse_loss(eps_pred, noise)
        return loss

    @torch.no_grad()
    def predict_action(self, obs_dict):
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B = nobs.shape[0]
        To = self.n_obs_steps

        global_cond = nobs[:, :To, :].reshape(B, -1)

        shape = (B, self.horizon, self.action_dim)
        naction_pred = self.conditional_sample(shape, global_cond)

        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        start = To
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        return action
