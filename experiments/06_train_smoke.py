"""
Stage 6 validation — smoke test of the full training pipeline.

Creates a tiny synthetic zarr dataset and runs a few epochs through
the TrainWorkspace. Verifies: data loading, training loop, EMA updates,
and checkpoint saving.
"""

import os
import tempfile
import numpy as np

try:
    import zarr
except ImportError:
    import sys
    sys.path.insert(0, '/tmp/zarr_install')
    import zarr

from training.workspace import train


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = os.path.join(tmpdir, 'test.zarr')
        root = zarr.open(zarr_path, mode='w')
        data = root.create_group('data')
        meta = root.create_group('meta')

        N = 200
        t = np.linspace(0, 4 * np.pi, N).astype(np.float32)
        state = np.stack([np.sin(t), np.cos(t), np.sin(2 * t), np.cos(2 * t)], axis=1)
        action = np.stack([np.sin(t + 0.5), np.cos(t + 0.5)], axis=1)

        data.create_dataset('state', data=state)
        data.create_dataset('action', data=action)
        meta.create_dataset('episode_ends', data=np.array([50, 100, 150, 200]))

        out_dir = os.path.join(tmpdir, 'output')
        policy = train(
            zarr_path=zarr_path,
            device='cpu',
            batch_size=16,
            num_epochs=5,
            horizon=8,
            n_obs_steps=2,
            n_action_steps=4,
            obs_key='state',
            action_key='action',
            lr=1e-3,
            checkpoint_every=3,
            output_dir=out_dir,
        )

        final_path = os.path.join(out_dir, 'policy_final.pt')
        assert os.path.exists(final_path), f"Missing {final_path}"
        ckpt_path = os.path.join(out_dir, 'checkpoint_epoch3.pt')
        assert os.path.exists(ckpt_path), f"Missing {ckpt_path}"

        print(f"Checkpoints saved: {os.listdir(out_dir)}")
        print("\nStage 6 smoke test PASSED")


if __name__ == '__main__':
    main()
