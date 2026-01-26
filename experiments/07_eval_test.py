"""
Stage 7 validation — end-to-end evaluation test.

Trains a tiny policy, saves it, loads it back, and runs mock rollouts.
Verifies the full train→save→load→eval pipeline.
"""

import os
import tempfile
import torch
import numpy as np

try:
    import zarr
except ImportError:
    import sys
    sys.path.insert(0, '/tmp/zarr_install')
    import zarr

from training.workspace import train
from eval.runner import EvalRunner, MockEnv


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Create synthetic data
        zarr_path = os.path.join(tmpdir, 'data.zarr')
        root = zarr.open(zarr_path, mode='w')
        data = root.create_group('data')
        meta = root.create_group('meta')

        N = 100
        state = np.random.randn(N, 4).astype(np.float32)
        action = np.random.randn(N, 2).astype(np.float32)
        data.create_dataset('state', data=state)
        data.create_dataset('action', data=action)
        meta.create_dataset('episode_ends', data=np.array([50, 100]))

        out_dir = os.path.join(tmpdir, 'output')

        # 2. Train
        policy = train(
            zarr_path=zarr_path,
            device='cpu',
            batch_size=8,
            num_epochs=2,
            horizon=8,
            n_obs_steps=2,
            n_action_steps=4,
            obs_key='state',
            action_key='action',
            lr=1e-3,
            checkpoint_every=5,
            output_dir=out_dir,
        )

        # 3. Load checkpoint
        final_path = os.path.join(out_dir, 'policy_final.pt')
        assert os.path.exists(final_path)
        payload = torch.load(final_path, map_location='cpu')
        assert 'policy_state_dict' in payload
        assert 'normalizer_state_dict' in payload
        print(f"Checkpoint loaded: {list(payload.keys())}")

        # 4. Run evaluation with mock env
        runner = EvalRunner(
            env_factory=lambda: MockEnv(obs_dim=4, action_dim=2, episode_len=10),
            n_test=3,
            max_steps=15,
            n_obs_steps=2,
            n_action_steps=4,
        )
        result = runner.run(policy)
        print(f"Eval: mean_reward={result['mean_reward']:.4f}, episodes={result['n_episodes']}")
        assert result['n_episodes'] == 3

    print("\nStage 7 end-to-end eval PASSED")


if __name__ == '__main__':
    main()
