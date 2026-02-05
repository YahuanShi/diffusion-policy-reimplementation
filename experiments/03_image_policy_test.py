"""
Stage 5 validation — tests DiffusionUnetImagePolicy end-to-end.

Verifies compute_loss, predict_action, gradient flow through encoder+diffusion,
and a short training loop on synthetic image→action data.
"""

import torch
from diffusion_policy.model.diffusion.scheduler import DDPMScheduler
from diffusion_policy.model.vision.encoder import MultiImageObsEncoder
from diffusion_policy.dataset.normalizer import LinearNormalizer
from diffusion_policy.policy.image import DiffusionUnetImagePolicy


def make_policy():
    shape_meta = {
        'obs': {
            'image': {'shape': (3, 64, 64), 'type': 'rgb'},
            'agent_pos': {'shape': (2,), 'type': 'low_dim'},
        },
        'action': {'shape': (2,)},
    }
    encoder = MultiImageObsEncoder(shape_meta, use_group_norm=True)
    scheduler = DDPMScheduler(num_train_timesteps=20)
    policy = DiffusionUnetImagePolicy(
        obs_encoder=encoder,
        noise_scheduler=scheduler,
        shape_meta=shape_meta,
        horizon=8,
        n_obs_steps=2,
        n_action_steps=4,
        num_inference_steps=5,
        diffusion_step_embed_dim=32,
        down_dims=[32, 64],
    )
    normalizer = LinearNormalizer()
    normalizer.fit({'action': torch.randn(50, 2)})
    policy.set_normalizer(normalizer)
    return policy


def test_forward():
    policy = make_policy()
    batch = {
        'image': torch.randn(2, 8, 3, 64, 64),
        'agent_pos': torch.randn(2, 8, 2),
        'action': torch.randn(2, 8, 2),
    }
    loss = policy.compute_loss(batch)
    assert loss.dim() == 0
    assert loss.item() > 0
    print(f"compute_loss: {loss.item():.4f} OK")


def test_inference():
    policy = make_policy()
    policy.eval()
    obs = {
        'image': torch.randn(2, 2, 3, 64, 64),
        'agent_pos': torch.randn(2, 2, 2),
    }
    action = policy.predict_action(obs)
    assert action.shape == (2, 4, 2)
    print(f"predict_action: {action.shape} OK")


def test_gradient_flow():
    policy = make_policy()
    batch = {
        'image': torch.randn(2, 8, 3, 64, 64),
        'agent_pos': torch.randn(2, 8, 2),
        'action': torch.randn(2, 8, 2),
    }
    loss = policy.compute_loss(batch)
    loss.backward()

    encoder_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in policy.obs_encoder.parameters()
    )
    unet_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in policy.model.parameters()
    )
    assert encoder_has_grad, "No gradient in encoder"
    assert unet_has_grad, "No gradient in U-Net"
    print("gradient flow: encoder + U-Net OK")


def test_training():
    torch.manual_seed(42)
    policy = make_policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    batch = {
        'image': torch.randn(4, 8, 3, 64, 64),
        'agent_pos': torch.randn(4, 8, 2),
        'action': torch.randn(4, 8, 2),
    }

    losses = []
    for i in range(20):
        loss = policy.compute_loss(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
        if i % 5 == 0:
            print(f"  train step {i}: loss={loss.item():.4f}")

    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print(f"training: {losses[0]:.4f} -> {losses[-1]:.4f} OK")


def main():
    test_forward()
    test_inference()
    test_gradient_flow()
    test_training()
    print("\nStage 5 validation PASSED")


if __name__ == '__main__':
    main()
