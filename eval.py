"""
Evaluation entry point — loads a checkpoint and runs PushT rollouts.

Usage:
    uv run python eval.py --checkpoint checkpoints/best.pt
"""

import argparse
import torch
import json
from eval.runner import PushTRunner

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--n_test', type=int, default=50)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location=args.device)
    policy = payload['ema_policy']
    policy.eval()

    runner = PushTRunner(n_test=args.n_test, device=args.device)
    result = runner.run(policy)

    print(f"mean_score: {result['mean_score']:.4f}")
    print(json.dumps(result, indent=2))
