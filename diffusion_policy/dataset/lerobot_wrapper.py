"""
LeRobot dataset wrapper — bridges LeRobot format to DiffusionUnetImagePolicy.

LeRobot format:
  - parquet for low-dim data (state, action), mp4 for video
  - delta_timestamps for temporal window sampling (replaces hand-written SequenceSampler)
  - Supports Hub datasets (repo_id) and local datasets (root)

This wrapper does three things:
  1. Key renaming: observation.image → observation_image (nn.ModuleDict forbids dots in names)
  2. Auto-build shape_meta: infer the policy config from LeRobot features
  3. Normalizer: build action + state normalizer from dataset statistics
"""

import torch
from torch.utils.data import Dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from diffusion_policy.dataset.normalizer import LinearNormalizer


def _safe_key(k):
    """observation.images.cam → observation_images_cam (ModuleDict forbids dots in key names)"""
    return k.replace(".", "_")


class LeRobotImageDataset(Dataset):
    """Wraps LeRobotDataset, output format directly compatible with DiffusionUnetImagePolicy.compute_loss()"""

    def __init__(
        self,
        repo_id,
        root=None,
        episodes=None,
        horizon=16,
        n_obs_steps=2,
        image_keys=None,
        state_key="observation.state",
        action_key="action",
        video_backend="pyav",
        image_transforms=None,
    ):
        meta_ds = LeRobotDataset(
            repo_id=repo_id, root=root, episodes=episodes, video_backend=video_backend
        )
        fps = meta_ds.fps
        features = meta_ds.features

        if image_keys is None:
            image_keys = sorted(
                k
                for k, v in features.items()
                if v.get("dtype") == "video"
                or (
                    isinstance(v.get("shape", ()), (list, tuple))
                    and len(v["shape"]) == 3
                )
            )

        ts = [i / fps for i in range(horizon)]
        dt = {action_key: ts}
        for k in image_keys:
            dt[k] = ts
        if state_key in features:
            dt[state_key] = ts

        del meta_ds

        self.ds = LeRobotDataset(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            video_backend=video_backend,
            image_transforms=image_transforms,
            delta_timestamps=dt,
        )

        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.image_keys = image_keys
        self.state_key = state_key
        self.action_key = action_key
        self.obs_keys = image_keys + ([state_key] if state_key in features else [])
        self._shape_meta = self._build_shape_meta()

    def _build_shape_meta(self):
        obs_meta = {}
        for key in self.image_keys:
            feat = self.ds.features[key]
            shape = tuple(feat["shape"])
            if len(shape) == 3:
                h, w, c = shape
                obs_meta[_safe_key(key)] = {"shape": (c, h, w), "type": "rgb"}
            else:
                obs_meta[_safe_key(key)] = {"shape": shape, "type": "rgb"}

        if self.state_key in self.ds.features:
            feat = self.ds.features[self.state_key]
            obs_meta[_safe_key(self.state_key)] = {
                "shape": tuple(feat["shape"]),
                "type": "low_dim",
            }

        action_feat = self.ds.features[self.action_key]
        return {
            "obs": obs_meta,
            "action": {"shape": tuple(action_feat["shape"])},
        }

    @property
    def shape_meta(self):
        return self._shape_meta

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        out = {}
        for key in self.obs_keys:
            out[_safe_key(key)] = item[key]
        out["action"] = item[self.action_key]
        return out

    def get_normalizer(self) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        hf = self.ds.hf_dataset

        # Column access is orders of magnitude faster than row-by-row iteration
        stats = {"action": torch.tensor(hf[self.action_key])}
        if self.state_key in self.ds.features:
            stats[_safe_key(self.state_key)] = torch.tensor(hf[self.state_key])

        normalizer.fit(stats)
        return normalizer
