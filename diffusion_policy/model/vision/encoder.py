"""
Multi-image observation encoder — encodes images + low-dim observations into a fixed-dim feature vector.

Architecture:
  Each RGB image -> ResNet18 (fc removed) -> 512-dim feature
  Low-dim data (state) -> concatenated directly
  All features concatenated -> obs_feature_dim

Key design choices:
  - GroupNorm replaces BatchNorm: robot data has small batches (4~64), BN statistics are unstable
  - share_rgb_model: multiple cameras share one ResNet backbone (saves params, may reduce accuracy)
  - No ImageNet pretrained weights: robot images differ too much from ImageNet, training from scratch is better
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF


def get_resnet(name="resnet18", weights=None):
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights)
    resnet.fc = nn.Identity()
    return resnet


def replace_submodules(root_module, predicate, func):
    """Replace submodules matching predicate. Used for BatchNorm → GroupNorm swap."""
    if predicate(root_module):
        return func(root_module)
    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    return root_module


class RandomCenterCrop(nn.Module):
    """Random crop during training, center crop during eval — no behaviour change at inference."""

    def __init__(self, size):
        super().__init__()
        self.size = size  # (h, w)

    def forward(self, x):
        h, w = self.size
        if self.training:
            i, j, th, tw = torchvision.transforms.RandomCrop.get_params(x, (h, w))
            return TF.crop(x, i, j, th, tw)
        else:
            return TF.center_crop(x, [h, w])


class MultiImageObsEncoder(nn.Module):
    def __init__(
        self,
        shape_meta,
        rgb_model_name="resnet18",
        rgb_model_weights=None,
        use_group_norm=True,
        share_rgb_model=False,
        imagenet_norm=False,
        resize_shape=None,
        crop_shape=None,
        random_crop=False,
    ):
        super().__init__()

        rgb_keys = []
        low_dim_keys = []
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = {}

        def _make_rgb_model():
            m = get_resnet(rgb_model_name, weights=rgb_model_weights)
            if use_group_norm:
                m = replace_submodules(
                    m,
                    predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                    func=lambda x: nn.GroupNorm(
                        num_groups=x.num_features // 16, num_channels=x.num_features
                    ),
                )
            return m

        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            obs_type = attr.get("type", "low_dim")
            key_shape_map[key] = shape

            if obs_type == "rgb":
                rgb_keys.append(key)
                if not share_rgb_model:
                    key_model_map[key] = _make_rgb_model()

                transforms = []
                if resize_shape is not None:
                    h, w = (
                        resize_shape
                        if not isinstance(resize_shape, dict)
                        else resize_shape[key]
                    )
                    transforms.append(torchvision.transforms.Resize(size=(h, w)))
                if crop_shape is not None:
                    h, w = (
                        crop_shape
                        if not isinstance(crop_shape, dict)
                        else crop_shape[key]
                    )
                    if random_crop:
                        transforms.append(RandomCenterCrop(size=(h, w)))
                    else:
                        transforms.append(
                            torchvision.transforms.CenterCrop(size=(h, w))
                        )
                if imagenet_norm:
                    transforms.append(
                        torchvision.transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        )
                    )
                key_transform_map[key] = (
                    nn.Sequential(*transforms) if transforms else nn.Identity()
                )

            elif obs_type == "low_dim":
                low_dim_keys.append(key)
            else:
                raise ValueError(f"Unknown obs type: {obs_type}")

        if share_rgb_model:
            key_model_map["rgb"] = _make_rgb_model()

        self.rgb_keys = sorted(rgb_keys)
        self.low_dim_keys = sorted(low_dim_keys)
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.key_shape_map = key_shape_map
        self.share_rgb_model = share_rgb_model
        self.shape_meta = shape_meta

    def forward(self, obs_dict):
        batch_size = None
        features = []

        if self.share_rgb_model:
            imgs = []
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                img = self.key_transform_map[key](img)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)
            feature = self.key_model_map["rgb"](imgs)
            feature = feature.reshape(-1, batch_size, *feature.shape[1:])
            feature = torch.moveaxis(feature, 0, 1)
            feature = feature.reshape(batch_size, -1)
            features.append(feature)
        else:
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                img = self.key_transform_map[key](img)
                feature = self.key_model_map[key](img)
                features.append(feature)

        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            features.append(data)

        return torch.cat(features, dim=-1)

    @torch.no_grad()
    def output_shape(self):
        example = {}
        for key, attr in self.shape_meta["obs"].items():
            shape = tuple(attr["shape"])
            example[key] = torch.zeros((1,) + shape)
        out = self.forward(example)
        return out.shape[1:]
