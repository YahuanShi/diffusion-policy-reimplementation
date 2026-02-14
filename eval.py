"""
Evaluation entry point — loads a trained image policy checkpoint and evaluates it.

The checkpoint contains shape_meta, normalizer and the training config, so the model
architecture is fully reconstructed without hardcoding dimensions.

Three modes:
  offline (default) — predicted vs recorded action chunks on held-out LeRobot episodes.
                      Defaults to the val episodes stored in the checkpoint (train.py --val_ratio).
  --sim TASK        — closed-loop rollouts (lift, can, square) in the shared Franka
                      simulation (robot-sim repo, robosuite) with fixed seeds; reports the
                      success rate. Needs `uv pip install --python DP/bin/python -e ../robot-sim`.
  --mock            — rollout loop against a random-observation env (pipeline smoke test only;
                      the reward is meaningless).

Usage:
    python eval.py --checkpoint outputs/policy_final.pt --device cuda
    python eval.py --checkpoint outputs/policy_final.pt --device cuda --episodes 3 17 42
    python eval.py --checkpoint outputs/lift/policy_final.pt --device cuda --sim lift \
        --n_episodes 50 --video_dir outputs/lift/videos
    python eval.py --checkpoint outputs/policy_final.pt --mock
"""

import argparse
import json

from torch.utils.data import DataLoader

from diffusion_policy.policy.loader import load_policy
from evaluation.offline import evaluate_offline
from evaluation.runner import EvalRunner, MockEnv


def run_offline(policy, config, args):
    from diffusion_policy.dataset.lerobot_wrapper import LeRobotImageDataset

    repo_id = args.repo_id or config.get("repo_id")
    root = args.root or config.get("root")
    if repo_id is None:
        raise SystemExit("No repo_id in checkpoint config — pass --repo_id / --root")

    episodes = args.episodes or config.get("val_episodes")
    if not episodes:
        raise SystemExit(
            "Checkpoint has no held-out val episodes (trained with --val_ratio 0). "
            "Pass --episodes explicitly; note that scoring training episodes "
            "only measures fit, not generalisation."
        )

    dataset = LeRobotImageDataset(
        repo_id=repo_id,
        root=root,
        episodes=episodes,
        horizon=policy.horizon,
        n_obs_steps=policy.n_obs_steps,
        image_keys=config.get("image_keys"),
        state_key=config.get("state_key", "observation.state"),
        video_backend=config.get("video_backend", "pyav"),
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    print(f"Offline eval on episodes {episodes} ({len(dataset)} samples)")

    metrics = evaluate_offline(
        policy, loader, args.device, max_batches=args.max_batches
    )
    print(f"denoising loss : {metrics['loss']:.4f}")
    print(f"action MSE     : {metrics['action_mse']:.4f}")
    print(f"action MAE     : {metrics['action_mae']:.4f}")
    print(
        "MAE per dim    : "
        + " ".join(f"{v:.3f}" for v in metrics["action_mae_per_dim"])
    )
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"episodes": episodes, **metrics}, f, indent=2)
        print(f"saved {args.output_json}")


def run_mock(policy, shape_meta, args):
    runner = EvalRunner(
        env_factory=lambda: MockEnv(shape_meta=shape_meta),
        shape_meta=shape_meta,
        n_episodes=args.n_episodes,
        n_obs_steps=policy.n_obs_steps,
        n_action_steps=policy.n_action_steps,
    )
    result = runner.run(policy)
    print(f"mean_reward: {result['mean_reward']:.4f} +/- {result['std_reward']:.4f}")
    print(f"max_reward: {result['max_reward']:.4f}")


def run_sim(policy, shape_meta, args):
    from robot_sim.evaluate import evaluate

    from evaluation.sim_policy import DiffusionSimPolicy

    action_dim = shape_meta["action"]["shape"][0]
    action_space = {10: "abs", 7: "delta"}.get(action_dim)
    if action_space is None:
        raise SystemExit(f"Action dim {action_dim} matches no robot_sim action space")
    image_shapes = [
        v["shape"] for v in shape_meta["obs"].values() if v.get("type") == "rgb"
    ]
    image_size = image_shapes[0][-1]
    sim_policy = DiffusionSimPolicy(policy, shape_meta, device=args.device)
    evaluate(
        sim_policy,
        task=args.sim,
        n_episodes=args.n_episodes,
        start_seed=args.start_seed,
        image_size=image_size,
        action_space=action_space,
        max_steps=args.max_steps,
        video_dir=args.video_dir,
        n_videos=args.n_videos,
        output_json=args.output_json,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num_inference_steps", type=int, default=16)
    parser.add_argument("--mock", action="store_true", help="Random-obs smoke test")

    # Simulation mode
    parser.add_argument(
        "--sim", default=None, metavar="TASK", help="robot_sim task (lift, can, square)"
    )
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument(
        "--start_seed", type=int, default=100_000, help="Episode i uses start_seed + i"
    )
    parser.add_argument(
        "--max_steps", type=int, default=None, help="Episode length cap (task default)"
    )
    parser.add_argument(
        "--video_dir", default=None, help="Save videos of the first episodes"
    )
    parser.add_argument("--n_videos", type=int, default=5)

    # Offline mode
    parser.add_argument("--repo_id", default=None, help="Override checkpoint repo_id")
    parser.add_argument("--root", default=None, help="Override checkpoint dataset root")
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--output_json", default=None, help="Save metrics to JSON")
    args = parser.parse_args()

    policy, shape_meta, config = load_policy(
        args.checkpoint, args.device, num_inference_steps=args.num_inference_steps
    )

    if args.sim:
        run_sim(policy, shape_meta, args)
    elif args.mock:
        run_mock(policy, shape_meta, args)
    else:
        run_offline(policy, config, args)


if __name__ == "__main__":
    main()
