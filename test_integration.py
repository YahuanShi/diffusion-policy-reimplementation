"""
Integration test — run this before launching full training (Stage 8).
Verifies the full pipeline end-to-end using fake data. Takes < 30 seconds.

Pass: prints "All integration tests passed."
Fail: raises an exception with the specific failure point.
"""

import torch
from diffusion_policy.model.diffusion.scheduler import DDPMScheduler
from diffusion_policy.model.diffusion.unet1d import ConditionalUnet1D
from diffusion_policy.model.vision.encoder import MultiImageObsEncoder
from diffusion_policy.dataset.normalizer import LinearNormalizer
from diffusion_policy.policy.image import DiffusionUnetImagePolicy

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
B, TO, H = 4, 2, 16   # batch, obs steps, horizon


def main():
    # Build components
    obs_encoder = MultiImageObsEncoder(n_obs_steps=TO).to(DEVICE)
    model = ConditionalUnet1D(
        input_dim=2,
        global_cond_dim=obs_encoder.output_dim,
        down_dims=[64, 128, 256],   # small for fast test
    ).to(DEVICE)
    scheduler = DDPMScheduler(num_train_timesteps=100)
    policy = DiffusionUnetImagePolicy(
        obs_encoder=obs_encoder, model=model, noise_scheduler=scheduler,
        horizon=H, action_dim=2, n_obs_steps=TO, n_action_steps=8,
    ).to(DEVICE)

    # Fake batch
    batch = {
        'image':     torch.randn(B, TO, 3, 96, 96).to(DEVICE),
        'agent_pos': torch.randn(B, TO, 2).to(DEVICE),
        'action':    torch.randn(B, H, 2).to(DEVICE),
    }

    # Fit dummy normalizer
    normalizer = LinearNormalizer()
    normalizer.fit({
        'image':     batch['image'].reshape(-1, 3, 96, 96).cpu(),
        'agent_pos': batch['agent_pos'].reshape(-1, 2).cpu(),
        'action':    batch['action'].reshape(-1, 2).cpu(),
    })
    policy.set_normalizer(normalizer)

    # Test training step
    policy.train()
    loss = policy.compute_loss(batch)
    loss.backward()
    print(f"compute_loss        OK  loss={loss.item():.4f}")

    # Test inference
    policy.eval()
    obs_dict = {'image': batch['image'][:1], 'agent_pos': batch['agent_pos'][:1]}
    with torch.no_grad():
        actions = policy.predict_action(obs_dict)
    assert actions.shape == (1, 8, 2), f"Expected (1,8,2), got {actions.shape}"
    print(f"predict_action      OK  shape={tuple(actions.shape)}")

    print("\nAll integration tests passed.")


if __name__ == '__main__':
    main()
