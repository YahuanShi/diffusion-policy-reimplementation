# Stage 6 — how-to-code.md
# Exponential Moving Average of model weights for stable inference.
# Reference: diffusion_policy-main/diffusion_policy/model/diffusion/ema_model.py


class EMAModel:
    def __init__(self, parameters, power=0.75, max_value=0.9999):
        self.shadow_params = ...
        self.power = power
        self.max_value = max_value
        self.step_count = 0

    def get_decay(self):
        # Returns decay value that ramps from 0 → max_value as step_count grows
        ...

    def step(self, parameters):
        # Update shadow weights: shadow = decay * shadow + (1 - decay) * param
        ...

    def copy_to(self, parameters):
        # Copy shadow weights into model parameters for eval
        ...
