"""
Stage 3 validation — trains the lowdim diffusion policy on synthetic data.

Creates a deterministic obs→action mapping and verifies the policy can learn it.
Success criterion: training loss drops steadily over 50 epochs.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from diffusion_policy.model.diffusion.scheduler import DDPMScheduler
from diffusion_policy.model.diffusion.unet1d import ConditionalUnet1D
from diffusion_policy.dataset.normalizer import LinearNormalizer
from diffusion_policy.policy.lowdim import DiffusionUnetLowdimPolicy


OBS_DIM = 4
ACTION_DIM = 2
HORIZON = 16
N_OBS_STEPS = 2
N_ACTION_STEPS = 8


def make_synthetic_dataset(n_episodes=50, ep_len=32):
    obs_list, action_list = [], []
    for _ in range(n_episodes):
        phase = torch.rand(1) * 2 * 3.14159
        t = torch.linspace(0, 2 * 3.14159, ep_len).unsqueeze(-1)
        obs = torch.cat([torch.sin(t + phase), torch.cos(t + phase),
                         torch.sin(2 * t + phase), torch.cos(2 * t + phase)], dim=-1)
        action = torch.cat([torch.sin(t + phase + 0.5),
                            torch.cos(t + phase + 0.5)], dim=-1)
        obs_list.append(obs)
        action_list.append(action)

    all_obs = torch.cat(obs_list, dim=0)
    all_action = torch.cat(action_list, dim=0)

    batched_obs = all_obs.unfold(0, HORIZON, 1).permute(0, 2, 1)[:n_episodes * (ep_len - HORIZON + 1)]
    batched_action = all_action.unfold(0, HORIZON, 1).permute(0, 2, 1)[:n_episodes * (ep_len - HORIZON + 1)]

    # Trim to multiple of batch size
    n = (len(batched_obs) // 32) * 32
    return batched_obs[:n], batched_action[:n]


def main():
    obs_data, action_data = make_synthetic_dataset()
    print(f"Dataset: {obs_data.shape[0]} windows, obs={obs_data.shape}, action={action_data.shape}")

    normalizer = LinearNormalizer()
    normalizer.fit({
        'obs': obs_data.reshape(-1, OBS_DIM),
        'action': action_data.reshape(-1, ACTION_DIM),
    })

    model = ConditionalUnet1D(
        input_dim=ACTION_DIM,
        global_cond_dim=OBS_DIM * N_OBS_STEPS,
        down_dims=[64, 128],
        diffusion_step_embed_dim=64,
    )
    scheduler = DDPMScheduler(num_train_timesteps=100)
    policy = DiffusionUnetLowdimPolicy(
        model=model,
        noise_scheduler=scheduler,
        horizon=HORIZON,
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        n_obs_steps=N_OBS_STEPS,
        n_action_steps=N_ACTION_STEPS,
        num_inference_steps=10,
    )
    policy.set_normalizer(normalizer)

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Model params: {n_params:,}")

    dataset = TensorDataset(obs_data, action_data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3, weight_decay=1e-6)

    losses_history = []
    for epoch in range(50):
        epoch_losses = []
        for obs_batch, action_batch in dataloader:
            batch = {'obs': obs_batch, 'action': action_batch}
            loss = policy.compute_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            epoch_losses.append(loss.item())
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses_history.append(avg_loss)
        if epoch % 10 == 0 or epoch == 49:
            print(f"epoch {epoch:>3}: loss={avg_loss:.4f}")

    # Verify loss decreased
    first_5 = sum(losses_history[:5]) / 5
    last_5 = sum(losses_history[-5:]) / 5
    print(f"\nFirst 5 avg: {first_5:.4f}, Last 5 avg: {last_5:.4f}")
    assert last_5 < first_5 * 0.8, f"Loss did not decrease enough: {first_5:.4f} -> {last_5:.4f}"

    # Test inference
    policy.eval()
    test_obs = obs_data[:4, :N_OBS_STEPS, :]
    action = policy.predict_action({'obs': test_obs})
    print(f"predict_action shape: {action.shape}")
    assert action.shape == (4, N_ACTION_STEPS, ACTION_DIM)
    print(f"action range: [{action.min().item():.2f}, {action.max().item():.2f}]")

    print("\nStage 3 validation PASSED")


if __name__ == '__main__':
    main()
