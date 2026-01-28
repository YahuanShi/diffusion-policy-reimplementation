# DDPM forward and reverse process.

import torch
from types import SimpleNamespace

class DDPMScheduler:
    def __init__(self, num_train_timesteps=100, beta_start=0.0001, beta_end=0.02, beta_schedule='squaredcos_cap_v2'):
        self.num_train_timesteps = num_train_timesteps

        if beta_schedule == 'linear':
            # T evenly-spaced values from beta_start to beta_end
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps).float()

        elif beta_schedule == 'squaredcos_cap_v2':
            # Cosine schedule (Nichol & Dhariwal 2021) - more stable than linear
            # Formula:  ᾱ_t = cos²( (t/T + s) / (1+s) · π/2 ) where s=0.008
            # Then β_t = 1 - ᾱ_t / ᾱ_{t-1}, clipped to [0, 0.999]
            s = 0.008
            steps = num_train_timesteps + 1
            t = torch.linspace(0, num_train_timesteps, steps, dtype=torch.float64)
            alphas_cumprod = torch.cos((t / num_train_timesteps + s) / (1 + s) * torch.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0] # normalize so ᾱ_0 = 1
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            self.betas = torch.clip(betas, 0, 0.999).float()

        # Everything below is schedule-independent — computed from self.betas
        # α_t = 1 - β_t
        self.alphas = 1.0 - self.betas

        # ᾱ_t = cumulative product of α_1 ... α_t
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Precompute the two square root terms used in add_noise
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # ᾱ_{t-1} for posterior variance computation (ᾱ_{-1} = 1 by convention)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        # β̃_t = β_t · (1 - ᾱ_{t-1}) / (1 - ᾱ_t)  — "fixed small" posterior variance
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_variance[0] = 0.0  # no noise at t=0

        self.timesteps = torch.arange(num_train_timesteps - 1, -1, -1)

    def add_noise(self, x0, noise, timesteps):
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].to(x0.device)
        sqrt_one_minus_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].to(x0.device)

        while sqrt_alpha_prod.dim() < x0.dim():
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_prod = sqrt_one_minus_prod.unsqueeze(-1)

        return sqrt_alpha_prod * x0 + sqrt_one_minus_prod * noise
    
    def step(self, eps_pred, t, x_t, clip_sample=True, clip_range=1.0):
        device = x_t.device
        alpha_prod_t = self.alphas_cumprod[t].to(device)
        alpha_prod_t_prev = self.alphas_cumprod_prev[t].to(device)
        beta_prod_t = 1.0 - alpha_prod_t

        pred_x0 = (x_t - beta_prod_t.sqrt() * eps_pred) / alpha_prod_t.sqrt()

        if clip_sample:
            pred_x0 = pred_x0.clamp(-clip_range, clip_range)

        pred_x0_coeff = alpha_prod_t_prev.sqrt() * self.betas[t].to(device) / beta_prod_t
        current_sample_coeff = self.alphas[t].to(device).sqrt() * (1.0 - alpha_prod_t_prev) / beta_prod_t
        pred_prev_mean = pred_x0_coeff * pred_x0 + current_sample_coeff * x_t

        if t > 0:
            variance = self.posterior_variance[t].to(device).sqrt() * torch.randn_like(x_t)
        else:
            variance = torch.zeros_like(x_t)

        prev_sample = pred_prev_mean + variance
        return SimpleNamespace(prev_sample=prev_sample)

    def set_timesteps(self, num_inference_steps):
        # Called before the inference loop to set how many denoising steps to run
        # For DDPM: evenly spaced from T-1 down to 0
        self.timesteps = torch.arange(num_inference_steps - 1, -1, -1)

if __name__ == '__main__':
    scheduler = DDPMScheduler(num_train_timesteps = 100)
    print(f"betas_shape : {scheduler.betas.shape}")   # should be torch.Size([100])

    print(f"betas[0]    : {scheduler.betas[0]:.6f}")  # should be small (~0.0001)
    print(f"betas[-1]   : {scheduler.betas[-1]:.6f}") # should be larger (~0.02)
    print(f"all positive: {(scheduler.betas > 0).all()}") # Ture

    print(f"ᾱ_0 = {scheduler.alphas_cumprod[0]:.4f}")   # should be close to 1.0
    print(f"ᾱ_99 = {scheduler.alphas_cumprod[-1]:.4f}") # should be close to 0.0
    print(f"√ᾱ_0 = {scheduler.sqrt_alphas_cumprod[0]:.4f}") # close to 1.0
    print(f"√ᾱ_99 = {scheduler.sqrt_one_minus_alphas_cumprod[-1]:.4f}") # close to 1.0
    
    x0 = torch.zeros(4,16,2)        # clean signal = all zeros
    noise = torch.ones(4,16,2)      # noise = all ones

    xt_early = scheduler.add_noise(x0, noise, torch.tensor([0,0,0,0]))
    xt_late  = scheduler.add_noise(x0, noise, torch.tensor([99,99,99,99]))

    print(f"t=0 mean: {xt_early.mean():.4f}")   # close to 0.0 (signal dominates)
    print(f"t=99 mean: {xt_late.mean():.4f}")   # close to 1.0 (noise dominates)

    # A clean signal at x=1
    x0 = torch.ones(1,1,1)
    noise = torch.randn_like(x0)

    # Corrupt it to t=50
    t = torch.tensor([50])
    xt = scheduler.add_noise(x0, noise, t)

    # One reverse step with the TRUE noise (oracle, not a learned model)
    out = scheduler.step(noise, 50, xt, clip_sample=False)
    print(f"x0      = {x0.item():.4f}")
    print(f"x_{50}  = {xt.item():.4f}")
    print(f"x_{49}  = {out.prev_sample.item():.4f}") # should be closer to x0 than xt
