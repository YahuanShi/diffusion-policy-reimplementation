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

        self.timesteps = torch.arange(num_train_timesteps - 1, -1, -1)

    def add_noise(self, x0, noise, timesteps):
        # Index the precomputed values by timestep for each batch item
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]  # [B]
        sqrt_one_minus_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]  # [B]

        # Reshape [B] -> [B,1,1] so it broadcasts over [B,H,D]
        while sqrt_alpha_prod.dim() < x0.dim():
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_prod = sqrt_one_minus_prod.unsqueeze(-1)

        # x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
        return sqrt_alpha_prod * x0 + sqrt_one_minus_prod * noise
    
    def step(self, eps_pred, t, x_t):
        # Scalar values for this timestep
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        sqrt_one_minus_alphas_prod = self.sqrt_one_minus_alphas_cumprod[t]

        # Predicted clean signal x_0 estimate (rearranging the forward formula)
        # x_{t-1} mean (no noise term yet)
        pred_prev_mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / sqrt_one_minus_alphas_prod) *eps_pred)

        # Add variance (skip at t=0, no noise on the last step)
        if t > 0:
            variance = torch.sqrt(beta_t) * torch.randn_like(x_t)
        else:
            variance = torch.zeros_like(x_t)
        
        prev_sample = pred_prev_mean + variance

        # Return an object with .prev_sample to match the diffusers API
        return SimpleNamespace(prev_sample = prev_sample)

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

    # One reserse step with the TRUE noise (oracle, not a learned model)
    out = scheduler.step(noise, 50, xt)
    print(f"x0      = {x0.item():.4f}")
    print(f"x_{50}  = {xt.item():.4f}")
    print(f"x_{49}  = {out.prev_sample.item():.4f}") # should be closer to x0 than xt
