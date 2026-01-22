import copy
import torch


class EMAModel:
    def __init__(self, model, power=2/3, min_value=0.0, max_value=0.9999,
                 update_after_step=0, inv_gamma=1.0):
        self.averaged_model = copy.deepcopy(model)
        self.averaged_model.eval()
        self.averaged_model.requires_grad_(False)

        self.power = power
        self.min_value = min_value
        self.max_value = max_value
        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.optimization_step = 0
        self.decay = 0.0

    def get_decay(self, optimization_step):
        step = max(0, optimization_step - self.update_after_step - 1)
        if step <= 0:
            return 0.0
        value = 1 - (1 + step / self.inv_gamma) ** -self.power
        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model):
        self.decay = self.get_decay(self.optimization_step)
        for param, ema_param in zip(new_model.parameters(),
                                     self.averaged_model.parameters()):
            if not param.requires_grad:
                ema_param.copy_(param.data)
            else:
                ema_param.mul_(self.decay)
                ema_param.add_(param.data, alpha=1 - self.decay)
        self.optimization_step += 1

    def copy_to(self, model):
        for param, ema_param in zip(model.parameters(),
                                     self.averaged_model.parameters()):
            param.data.copy_(ema_param.data)

    def state_dict(self):
        return {
            'averaged_model': self.averaged_model.state_dict(),
            'optimization_step': self.optimization_step,
            'decay': self.decay,
        }

    def load_state_dict(self, state_dict):
        self.averaged_model.load_state_dict(state_dict['averaged_model'])
        self.optimization_step = state_dict['optimization_step']
        self.decay = state_dict['decay']
