"""
Convert HDF5 robot episodes to LeRobot v3 dataset format.

Usage:
    python scripts/hdf5_to_lerobot.py \
        --input /path/to/hdf5_dir \
        --repo_id local/my_dataset \
        --root /path/to/output \
        --fps 10

The HDF5 files should have structure:
    action: (T, action_dim)
    observations/eef_pose: (T, 6)
    observations/qpos: (T, 7)
    observations/qvel: (T, 7)
    observations/images/exterior_image_1_left: (T, H, W, 3)
    observations/images/wrist_image_left: (T, H, W, 3)
"""

import argparse
import os
import re
import shutil

import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def get_episode_files(input_dir):
    files = [f for f in os.listdir(input_dir) if f.endswith(".hdf5")]
    files.sort(key=lambda x: int(re.search(r"(\d+)", x).group(1)))
    return files


def inspect_hdf5(filepath):
    datasets = {}
    with h5py.File(filepath, "r") as f:

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets[name] = {"shape": obj.shape, "dtype": str(obj.dtype)}

        f.visititems(visitor)
    return datasets


def build_features(hdf5_info, state_keys, image_keys, action_key="action"):
    features = {}

    for img_key in image_keys:
        hdf5_key = f"observations/images/{img_key}"
        shape = hdf5_info[hdf5_key]["shape"][1:]  # drop T
        lerobot_key = f"observation.images.{img_key}"
        features[lerobot_key] = {
            "dtype": "video",
            "shape": tuple(shape),
            "names": ["height", "width", "channel"],
        }

    state_dim = sum(hdf5_info[f"observations/{k}"]["shape"][1] for k in state_keys)
    features["observation.state"] = {
        "dtype": "float32",
        "shape": (state_dim,),
        "names": {"motors": [f"motor_{i}" for i in range(state_dim)]},
    }

    action_dim = hdf5_info[action_key]["shape"][1]
    features["action"] = {
        "dtype": "float32",
        "shape": (action_dim,),
        "names": {"motors": [f"motor_{i}" for i in range(action_dim)]},
    }

    return features


def convert(
    input_dir,
    repo_id,
    root,
    fps,
    state_keys=("eef_pose", "qpos"),
    image_keys=("exterior_image_1_left", "wrist_image_left"),
    task_description="Pick and place",
    vcodec="auto",
):

    episode_files = get_episode_files(input_dir)
    print(f"Found {len(episode_files)} episodes")

    hdf5_info = inspect_hdf5(os.path.join(input_dir, episode_files[0]))
    print("HDF5 structure:")
    for k, v in hdf5_info.items():
        print(f"  {k}: {v}")

    features = build_features(hdf5_info, state_keys, image_keys)
    print(f"\nLeRobot features: {list(features.keys())}")

    output_path = Path(root)
    if output_path.exists():
        shutil.rmtree(output_path)

    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=output_path,
        features=features,
        use_videos=True,
        vcodec=vcodec,
    )

    for ep_idx, ep_file in enumerate(tqdm(episode_files, desc="Converting")):
        filepath = os.path.join(input_dir, ep_file)
        with h5py.File(filepath, "r") as f:
            actions = f["action"][:].astype(np.float32)
            states = np.concatenate(
                [f[f"observations/{k}"][:].astype(np.float32) for k in state_keys],
                axis=-1,
            )
            images = {k: f[f"observations/images/{k}"][:] for k in image_keys}
            T = actions.shape[0]

        for t in range(T):
            frame = {
                "action": actions[t],
                "observation.state": states[t],
                "task": task_description,
            }
            for img_key in image_keys:
                frame[f"observation.images.{img_key}"] = images[img_key][t]

            ds.add_frame(frame)

        ds.save_episode()

        if (ep_idx + 1) % 50 == 0:
            print(f"  Converted {ep_idx + 1}/{len(episode_files)} episodes")

    ds.finalize()
    print(f"\nDone. Dataset saved to: {output_path}")
    print(f"  Total episodes: {ds.num_episodes}")
    print(f"  Total frames: {ds.num_frames}")
    print(f"  repo_id: {repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert HDF5 episodes to LeRobot format"
    )
    parser.add_argument(
        "--input", required=True, help="Directory containing episode_*.hdf5 files"
    )
    parser.add_argument("--repo_id", default="local/task_graph", help="LeRobot repo_id")
    parser.add_argument("--root", required=True, help="Output root directory")
    parser.add_argument("--fps", type=int, default=10, help="Dataset FPS")
    parser.add_argument(
        "--state_keys",
        nargs="+",
        default=["eef_pose", "qpos"],
        help="Observation keys to concat as state",
    )
    parser.add_argument(
        "--image_keys",
        nargs="+",
        default=["exterior_image_1_left", "wrist_image_left"],
        help="Image observation keys",
    )
    parser.add_argument("--task", default="Pick and place", help="Task description")
    parser.add_argument("--vcodec", default="auto", help="Video codec")
    args = parser.parse_args()

    convert(
        input_dir=args.input,
        repo_id=args.repo_id,
        root=args.root,
        fps=args.fps,
        state_keys=tuple(args.state_keys),
        image_keys=tuple(args.image_keys),
        task_description=args.task,
        vcodec=args.vcodec,
    )
