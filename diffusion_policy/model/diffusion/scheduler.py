"""
DDPM and DDIM schedulers — forward noise process and two reverse samplers.

Training always uses DDPM's forward process:
  x_t = sqrt(ᾱ_t) * x0 + sqrt(1 - ᾱ_t) * ε,  ε ~ N(0, I)
  Network learns ε_θ(x_t, t) to predict the noise; loss = MSE(ε_θ, ε)

Inference can use either sampler — both are compatible with the same trained weights:

  DDPMScheduler (stochastic, 100 steps):
    x_{t-1} = μ̃(x_t, ε_θ) + σ_t · z,  z ~ N(0, I)
    Adds noise at every step; requires T=100 denoising iterations.

  DDIMScheduler (deterministic, 16 steps):
    x_{t-1} = √ᾱ_{t-1} · x̂₀ + √(1-ᾱ_{t-1}) · ε_θ
    No stochastic noise term; 16 uniformly-spaced steps suffice → ~6× faster.
    Preferred for deployment (inference.py, eval.py).

References:
  DDPM — Ho et al. 2020 (https://arxiv.org/abs/2006.11239)
  DDIM — Song et al. 2020 (https://arxiv.org/abs/2010.02502)
"""

from types import SimpleNamespace

import torch


class DDPMScheduler:
    def __init__(
        self,
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
    ):
        self.num_train_timesteps = num_train_timesteps

        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_train_timesteps
            ).float()

        elif beta_schedule == "squaredcos_cap_v2":
            # Cosine schedule (Nichol & Dhariwal 2021), more stable than linear.
            # alpha_bar_t = cos^2((t/T + s)/(1+s) * pi/2), s=0.008 prevents alpha_bar_T = 0
            # Then beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}, clipped to [0, 0.999]
            # Advantage: noise is added more uniformly across timesteps
            s = 0.008
            steps = num_train_timesteps + 1
            t = torch.linspace(0, num_train_timesteps, steps, dtype=torch.float64)
            alphas_cumprod = (
                torch.cos((t / num_train_timesteps + s) / (1 + s) * torch.pi / 2) ** 2
            )
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            self.betas = torch.clip(betas, 0, 0.999).float()

        # ---- Everything below is derived from betas, schedule-independent ----

        # alpha_t = 1 - beta_t
        self.alphas = 1.0 - self.betas

        # alpha_bar_t = alpha_1 * alpha_2 * ... * alpha_t  (cumulative product)
        # Near 1 at t=0 (almost no noise), near 0 at t=T (pure noise)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Two sqrt coefficients for forward process: x_t = sqrt(a) * x0 + sqrt(1-a) * eps
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # alpha_bar_{t-1} for posterior variance (alpha_bar_{-1} = 1 by convention)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]), self.alphas_cumprod[:-1]]
        )

        # Posterior variance: beta_tilde_t = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        # This is the "fixed small" variance from the DDPM paper
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_variance[0] = 0.0  # no noise at t=0

        self.set_timesteps(num_train_timesteps)

    def add_noise(
        self, x0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Forward process: x_t = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * eps"""
        t_cpu = timesteps.cpu()
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[t_cpu].to(x0.device)
        sqrt_one_minus_prod = self.sqrt_one_minus_alphas_cumprod[t_cpu].to(x0.device)

        # Broadcast: timesteps is (B,), x0 is (B, H, D), need to unsqueeze
        while sqrt_alpha_prod.dim() < x0.dim():
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_prod = sqrt_one_minus_prod.unsqueeze(-1)

        return sqrt_alpha_prod * x0 + sqrt_one_minus_prod * noise

    def step(
        self,
        eps_pred: torch.Tensor,
        t: int,
        x_t: torch.Tensor,
        clip_sample: bool = True,
        clip_range: float = 1.0,
    ) -> SimpleNamespace:
        """
        One reverse (denoising) step: compute x_{t-1} from x_t and predicted noise.

        Uses "predict x0 then clip" strategy (key design in Diffusion Policy):
          1. Recover x0_hat = (x_t - sqrt(1-alpha_bar_t) * eps_pred) / sqrt(alpha_bar_t)
          2. Clip x0_hat to [-1, 1] (actions are normalized to this range)
          3. Compute posterior mean mu_tilde from clipped x0_hat and x_t
          4. Add posterior variance noise to get x_{t-1}

        Clipping is critical: prevents divergence during reverse process, especially early in training.
        """
        device = x_t.device
        t = int(t)
        alpha_prod_t = self.alphas_cumprod[t].to(device)
        # alpha_bar of the *next scheduled* timestep. Equals alphas_cumprod_prev[t]
        # when all T steps are used, but differs once set_timesteps() subsamples.
        alpha_prod_t_prev = self._alpha_prev[self._t_idx[t]].to(device)
        beta_prod_t = 1.0 - alpha_prod_t
        beta_prod_t_prev = 1.0 - alpha_prod_t_prev
        # Effective alpha/beta of the (possibly multi-step) jump t -> t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1.0 - current_alpha_t

        # Step 1: recover x0_hat from eps_pred
        pred_x0 = (x_t - beta_prod_t.sqrt() * eps_pred) / alpha_prod_t.sqrt()

        # Step 2: clip to normalized range
        if clip_sample:
            pred_x0 = pred_x0.clamp(-clip_range, clip_range)

        # Step 3: posterior mean = coeff1 * x0_hat + coeff2 * x_t
        pred_x0_coeff = alpha_prod_t_prev.sqrt() * current_beta_t / beta_prod_t
        current_sample_coeff = current_alpha_t.sqrt() * beta_prod_t_prev / beta_prod_t
        pred_prev_mean = pred_x0_coeff * pred_x0 + current_sample_coeff * x_t

        # Step 4: add posterior variance noise (none at the final step)
        if t > 0:
            variance = beta_prod_t_prev / beta_prod_t * current_beta_t
            noise = variance.sqrt() * torch.randn_like(x_t)
        else:
            noise = torch.zeros_like(x_t)

        prev_sample = pred_prev_mean + noise
        return SimpleNamespace(prev_sample=prev_sample)

    def set_timesteps(self, num_inference_steps):
        """Uniformly subsample T-1 → 0 over num_inference_steps steps."""
        T = self.num_train_timesteps
        self.timesteps = torch.linspace(T - 1, 0, num_inference_steps).long()
        # alpha_cumprod for the step *preceding* each scheduled t
        prev_ts = torch.cat([self.timesteps[1:], torch.zeros(1, dtype=torch.long)])
        self._alpha_prev = self.alphas_cumprod[prev_ts].clone()
        self._alpha_prev[-1] = (
            1.0  # alpha_{t=-1} = 1 by convention (fully clean signal)
        )
        self._t_idx = {int(t): i for i, t in enumerate(self.timesteps)}


class DDIMScheduler(DDPMScheduler):
    """
    DDIM (Song et al. 2020) — deterministic reverse process.

    Forward process is identical to DDPM; only the reverse step changes:
      x0_hat  = (x_t - sqrt(1-a_t) * eps) / sqrt(a_t)        (recover clean sample)
      x_{t-1} = sqrt(a_{t-1}) * x0_hat + sqrt(1-a_{t-1}) * eps  (deterministic, eta=0)

    No stochastic noise term → 16 steps suffice instead of 100, ~6x faster at inference.
    Fully compatible with DDPM checkpoints — training is unchanged.
    """

    def step(self, eps_pred, t, x_t, clip_sample=True, clip_range=1.0):
        device = x_t.device
        t_val = int(t)
        alpha_t = self.alphas_cumprod[t_val].to(device)
        alpha_t_prev = self._alpha_prev[self._t_idx[t_val]].to(device)

        pred_x0 = (x_t - (1 - alpha_t).sqrt() * eps_pred) / alpha_t.sqrt()
        if clip_sample:
            pred_x0 = pred_x0.clamp(-clip_range, clip_range)

        prev_sample = (
            alpha_t_prev.sqrt() * pred_x0 + (1 - alpha_t_prev).sqrt() * eps_pred
        )
        return SimpleNamespace(prev_sample=prev_sample)


if __name__ == "__main__":
    scheduler = DDPMScheduler(num_train_timesteps=100)
    print(f"betas_shape : {scheduler.betas.shape}")  # should be torch.Size([100])

    print(f"betas[0]    : {scheduler.betas[0]:.6f}")  # should be small (~0.0001)
    print(f"betas[-1]   : {scheduler.betas[-1]:.6f}")  # should be larger (~0.02)
    print(f"all positive: {(scheduler.betas > 0).all()}")  # Ture

    print(f"ᾱ_0 = {scheduler.alphas_cumprod[0]:.4f}")  # should be close to 1.0
    print(f"ᾱ_99 = {scheduler.alphas_cumprod[-1]:.4f}")  # should be close to 0.0
    print(f"√ᾱ_0 = {scheduler.sqrt_alphas_cumprod[0]:.4f}")  # close to 1.0
    print(f"√ᾱ_99 = {scheduler.sqrt_one_minus_alphas_cumprod[-1]:.4f}")  # close to 1.0

    x0 = torch.zeros(4, 16, 2)  # clean signal = all zeros
    noise = torch.ones(4, 16, 2)  # noise = all ones

    xt_early = scheduler.add_noise(x0, noise, torch.tensor([0, 0, 0, 0]))
    xt_late = scheduler.add_noise(x0, noise, torch.tensor([99, 99, 99, 99]))

    print(f"t=0 mean: {xt_early.mean():.4f}")  # close to 0.0 (signal dominates)
    print(f"t=99 mean: {xt_late.mean():.4f}")  # close to 1.0 (noise dominates)

    # A clean signal at x=1
    x0 = torch.ones(1, 1, 1)
    noise = torch.randn_like(x0)

    # Corrupt it to t=50
    t = torch.tensor([50])
    xt = scheduler.add_noise(x0, noise, t)

    # One reverse step with the TRUE noise (oracle, not a learned model)
    out = scheduler.step(noise, 50, xt, clip_sample=False)
    print(f"x0      = {x0.item():.4f}")
    print(f"x_{50}  = {xt.item():.4f}")
    print(f"x_{49}  = {out.prev_sample.item():.4f}")  # should be closer to x0 than xt
