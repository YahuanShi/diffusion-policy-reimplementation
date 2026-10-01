"""
Evaluation entry point — loads a trained image policy checkpoint and runs rollouts.

The checkpoint contains shape_meta, normalizer and policy_config (architecture + image
resize/crop), so the policy is rebuilt exactly as trained without hardcoding anything.
Step/epoch checkpoints are evaluated with their EMA weights.

Usage:
    python eval.py --checkpoint outputs/policy_final.pt --mock
    python eval.py --checkpoint outputs/policy_final.pt --device cuda --mock
"""

import argparse

from diffusion_policy.policy.checkpoint import load_policy
from evaluation.runner import EvalRunner, MockEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_test", type=int, default=50)
    parser.add_argument("--num_inference_steps", type=int, default=16)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Only for legacy checkpoints without policy_config",
    )
    parser.add_argument(
        "--crop",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Only for legacy checkpoints without policy_config",
    )
    args = parser.parse_args()

    policy, shape_meta, cfg = load_policy(
        args.checkpoint,
        args.device,
        num_inference_steps=args.num_inference_steps,
        resize_shape=args.resize,
        crop_shape=args.crop,
    )

    if args.mock:

        def env_factory():
            return MockEnv(shape_meta=shape_meta)
    else:
        raise NotImplementedError(
            "Real env evaluation requires a gym environment. Use --mock for testing."
        )

    runner = EvalRunner(
        env_factory=env_factory,
        shape_meta=shape_meta,
        n_test=args.n_test,
        n_obs_steps=cfg["n_obs_steps"],
        n_action_steps=cfg["n_action_steps"],
    )
    result = runner.run(policy)
    print(f"mean_reward: {result['mean_reward']:.4f} +/- {result['std_reward']:.4f}")
    print(f"max_reward: {result['max_reward']:.4f}")


if __name__ == "__main__":
    main()
