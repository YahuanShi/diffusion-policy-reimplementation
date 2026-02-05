"""
Integration test — verifies the full pipeline end-to-end using fake data.

Run: python -m pytest tests/ -v
"""

import torch
from diffusion_policy.model.diffusion.scheduler import DDPMScheduler
from diffusion_policy.model.vision.encoder import MultiImageObsEncoder
from diffusion_policy.dataset.normalizer import LinearNormalizer
from diffusion_policy.policy.image import DiffusionUnetImagePolicy
from diffusion_policy.model.diffusion.ema import EMAModel

B, TO, H = 4, 2, 16


def test_image_policy():
    shape_meta = {
        'obs': {
            'image': {'shape': (3, 64, 64), 'type': 'rgb'},
            'agent_pos': {'shape': (2,), 'type': 'low_dim'},
        },
        'action': {'shape': (2,)},
    }

    encoder = MultiImageObsEncoder(shape_meta, use_group_norm=True)
    scheduler = DDPMScheduler(num_train_timesteps=50)
    policy = DiffusionUnetImagePolicy(
        obs_encoder=encoder,
        noise_scheduler=scheduler,
        shape_meta=shape_meta,
        horizon=H,
        n_obs_steps=TO,
        n_action_steps=8,
        num_inference_steps=5,
        diffusion_step_embed_dim=32,
        down_dims=[32, 64],
    )

    normalizer = LinearNormalizer()
    normalizer.fit({'action': torch.randn(50, 2)})
    policy.set_normalizer(normalizer)

    batch = {
        'image': torch.randn(B, H, 3, 64, 64),
        'agent_pos': torch.randn(B, H, 2),
        'action': torch.randn(B, H, 2),
    }

    policy.train()
    loss = policy.compute_loss(batch)
    loss.backward()
    assert loss.item() > 0

    policy.eval()
    obs_dict = {'image': batch['image'][:1, :TO], 'agent_pos': batch['agent_pos'][:1, :TO]}
    with torch.no_grad():
        actions = policy.predict_action(obs_dict)
    assert actions.shape == (1, 8, 2)


def test_ema():
    model = torch.nn.Linear(4, 2)
    ema = EMAModel(model)
    for _ in range(10):
        with torch.no_grad():
            model.weight.add_(torch.randn_like(model.weight) * 0.1)
        ema.step(model)
    assert ema.optimization_step == 10
    assert ema.decay > 0


if __name__ == '__main__':
    test_image_policy()
    test_ema()
    print("All integration tests passed.")
