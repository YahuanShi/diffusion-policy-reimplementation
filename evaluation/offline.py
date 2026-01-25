"""
Offline evaluation — compare predicted action chunks against recorded demonstrations.

Without a simulator, this is the cheapest signal of whether the policy learned anything:
  for each sample (obs window → ground-truth action horizon) in held-out episodes,
  run the full reverse process and measure the error on the executed chunk
  action[To-1 : To-1+Ta] — exactly the slice the robot would execute.

Metrics:
  loss        — denoising MSE (same objective as training, random t / ε)
  action_mse  — MSE between predicted and recorded actions, in raw (unnormalized) units
  action_mae  — mean absolute error, per dim, in raw units (degrees for UR3e joints)

Caveat: low action error ≠ high success rate. Demonstrations are multimodal, so a
policy can pick a valid-but-different mode and still be scored as "wrong". Use this
for comparing checkpoints / catching regressions, not as a substitute for robot trials.
"""

import torch


@torch.no_grad()
def evaluate_offline(policy, dataloader, device, max_batches=None):
    """
    policy:     DiffusionUnetImagePolicy (any scheduler; switched to eval mode)
    dataloader: yields LeRobotImageDataset batches (obs keys + "action" of length horizon)
    Returns:    dict of scalar metrics + per-dim MAE list
    """
    was_training = policy.training
    policy.eval()
    To, Ta = policy.n_obs_steps, policy.n_action_steps

    loss_sum = 0.0
    n_batches = 0
    sq_err_sum = None
    abs_err_sum = None
    n_actions = 0

    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        loss_sum += policy.compute_loss(batch).item()
        n_batches += 1

        obs_dict = {k: v for k, v in batch.items() if k != "action"}
        pred = policy.predict_action(obs_dict)  # (B, Ta, D)
        gt = batch["action"][:, To - 1 : To - 1 + Ta]

        err = pred - gt
        sq = err.pow(2).sum(dim=(0, 1))
        ab = err.abs().sum(dim=(0, 1))
        sq_err_sum = sq if sq_err_sum is None else sq_err_sum + sq
        abs_err_sum = ab if abs_err_sum is None else abs_err_sum + ab
        n_actions += err.shape[0] * err.shape[1]

    if was_training:
        policy.train()

    if n_batches == 0:
        raise ValueError("evaluate_offline: dataloader produced no batches")

    mse_per_dim = sq_err_sum / n_actions
    mae_per_dim = abs_err_sum / n_actions
    return {
        "loss": loss_sum / n_batches,
        "action_mse": mse_per_dim.mean().item(),
        "action_mae": mae_per_dim.mean().item(),
        "action_mae_per_dim": mae_per_dim.cpu().tolist(),
        "n_samples": n_actions // Ta,
    }
