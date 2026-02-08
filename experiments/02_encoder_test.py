"""
Stage 4 validation — tests MultiImageObsEncoder.

Verifies: GroupNorm replacement, output shapes, shared/independent backbones,
multi-camera support, and low_dim passthrough.
"""

import torch
import torch.nn as nn
from diffusion_policy.model.vision.encoder import MultiImageObsEncoder


def test_single_camera():
    shape_meta = {
        "obs": {
            "image": {"shape": (3, 96, 96), "type": "rgb"},
            "agent_pos": {"shape": (2,), "type": "low_dim"},
        }
    }
    encoder = MultiImageObsEncoder(shape_meta, use_group_norm=True)

    bn = sum(1 for m in encoder.modules() if isinstance(m, nn.BatchNorm2d))
    gn = sum(1 for m in encoder.modules() if isinstance(m, nn.GroupNorm))
    assert bn == 0, f"Found {bn} BatchNorm2d (should be 0)"
    assert gn > 0, "No GroupNorm found"

    out_shape = encoder.output_shape()
    assert out_shape == torch.Size([514]), f"Expected (514,), got {out_shape}"

    obs = {"image": torch.randn(2, 3, 96, 96), "agent_pos": torch.randn(2, 2)}
    out = encoder(obs)
    assert out.shape == (2, 514)
    print("single camera: OK")


def test_multi_camera_independent():
    shape_meta = {
        "obs": {
            "cam_front": {"shape": (3, 84, 84), "type": "rgb"},
            "cam_side": {"shape": (3, 84, 84), "type": "rgb"},
            "state": {"shape": (7,), "type": "low_dim"},
        }
    }
    encoder = MultiImageObsEncoder(shape_meta, use_group_norm=True)
    out_shape = encoder.output_shape()
    assert out_shape == torch.Size([512 * 2 + 7])

    obs = {
        "cam_front": torch.randn(2, 3, 84, 84),
        "cam_side": torch.randn(2, 3, 84, 84),
        "state": torch.randn(2, 7),
    }
    out = encoder(obs)
    assert out.shape == (2, 1031)
    print("multi camera (independent): OK")


def test_shared_backbone():
    shape_meta = {
        "obs": {
            "cam_a": {"shape": (3, 96, 96), "type": "rgb"},
            "cam_b": {"shape": (3, 96, 96), "type": "rgb"},
        }
    }
    encoder = MultiImageObsEncoder(
        shape_meta, share_rgb_model=True, use_group_norm=True
    )
    out_shape = encoder.output_shape()
    assert out_shape == torch.Size([512 * 2])

    obs = {"cam_a": torch.randn(3, 3, 96, 96), "cam_b": torch.randn(3, 3, 96, 96)}
    out = encoder(obs)
    assert out.shape == (3, 1024)
    print("shared backbone: OK")


def test_imagenet_norm():
    shape_meta = {"obs": {"img": {"shape": (3, 64, 64), "type": "rgb"}}}
    encoder = MultiImageObsEncoder(shape_meta, imagenet_norm=True, use_group_norm=True)
    obs = {"img": torch.ones(1, 3, 64, 64) * 0.5}
    out = encoder(obs)
    assert out.shape[0] == 1
    print("imagenet norm: OK")


def test_gradient_flow():
    shape_meta = {
        "obs": {
            "image": {"shape": (3, 64, 64), "type": "rgb"},
            "pos": {"shape": (3,), "type": "low_dim"},
        }
    }
    encoder = MultiImageObsEncoder(shape_meta, use_group_norm=True)
    obs = {"image": torch.randn(2, 3, 64, 64), "pos": torch.randn(2, 3)}
    out = encoder(obs)
    loss = out.sum()
    loss.backward()

    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters()
    )
    assert has_grad, "No gradients flowing through encoder"
    print("gradient flow: OK")


def main():
    test_single_camera()
    test_multi_camera_independent()
    test_shared_backbone()
    test_imagenet_norm()
    test_gradient_flow()
    print("\nStage 4 validation PASSED")


if __name__ == "__main__":
    main()
