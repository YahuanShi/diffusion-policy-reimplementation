"""
Conditional 1D U-Net — the noise prediction network for Diffusion Policy.

Architecture:
  Input:  noisy action sequence x_t (B, horizon, action_dim) + conditioning
  Output: predicted noise eps_theta (B, horizon, action_dim)

Conditioning method (obs_as_global_cond, the Diffusion Policy default):
  - Observation features are flattened into a vector, concatenated with timestep embedding
  - Injected into every ResBlock via FiLM modulation (not concatenated to the input sequence)
  - Benefit: U-Net only processes the action sequence, no inpainting mask needed

Data flow:
  noisy_action (B,H,D) -> transpose to (B,D,H) -> encoder -> bottleneck -> decoder -> transpose back
                               ^ skip connections ^
  timestep t -> SinusoidalEmb -> MLP -+
  obs_features ────────────────────── +-> concat -> global_cond -> FiLM into every ResBlock
"""

from typing import Union
import math
import torch
import torch.nn as nn
import einops


class SinusoidalPosEmb(nn.Module):
    """Encode a scalar timestep t into a dim-dimensional vector. Same formula as Transformer positional encoding."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish"""

    def __init__(self, in_channels, out_channels,kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish()
        )

    def forward(self,x):
        return self.block(x)

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Conv1d with kernel_size=3, stride=2, padding=1
        # stride=2 halves the sequence length
        self.conv = nn.Conv1d(dim, dim, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class ConditionalResidualBlock1D(nn.Module):
    """
    Conditional residual block — the basic building block of the U-Net.

    Structure: Conv1d -> GroupNorm -> Mish -> [FiLM modulation] -> Conv1d -> GroupNorm -> Mish -> + residual

    FiLM (Feature-wise Linear Modulation) is the key to conditioning injection:
      - cond_predict_scale=False: out = out + bias           (bias only)
      - cond_predict_scale=True:  out = scale * out + bias   (scale + bias)
    Conditioning (timestep + obs) affects each channel of the feature map via FiLM.
    """
    def __init__(self, in_channels, out_channels, cond_dim,
                 kernel_size=3, n_groups=8, cond_predict_scale=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        cond_channels = out_channels * 2 if cond_predict_scale else out_channels
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        # cond -> FiLM parameters (bias, or scale+bias)
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
        )

        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        out = self.blocks[0](x)
        # cond: (B, cond_dim) -> embed: (B, channels, 1), broadcasts over sequence dim
        embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    """
    Conditional 1D U-Net. The core network of Diffusion Policy.

    Args:
      input_dim:      action dimension (e.g. 7 for joint angles)
      global_cond_dim: global conditioning dim = obs_feature_dim * n_obs_steps
      down_dims:      channel counts per encoder layer, e.g. [256, 512, 1024]

    Example: down_dims=[256,512,1024], input_dim=7, horizon=16
      encoder:    7->256 (H=16) -> 256->512 (H=8) -> 512->1024 (H=4)
      bottleneck: 1024->1024 (H=4)
      decoder:    1024->512 (H=8) -> 512->256 (H=16)
      final:      256->7 (H=16)
    """
    def __init__(self,
                 input_dim,
                 local_cond_dim=None,
                 global_cond_dim=None,
                 diffusion_step_embed_dim=256,
                 down_dims=[256, 512, 1024],
                 kernel_size=3,
                 n_groups=8,
                 cond_predict_scale=False):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        # Timestep encoding: scalar t -> sinusoidal -> MLP -> dsed-dim vector
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        # Global conditioning = timestep_embed || obs_features
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        self.local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            self.local_cond_encoder = nn.ModuleList([
                ConditionalResidualBlock1D(
                    local_cond_dim, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    local_cond_dim, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
            ])

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale),
        ])

        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out * 2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def forward(self,
                sample: torch.Tensor,
                timestep: Union[torch.Tensor, float, int],
                local_cond=None, global_cond=None, **kwargs):
        """
        Forward pass: predict noise eps_theta(x_t, t, cond).

        Data flow:
          1. Transpose: (B, horizon, D) -> (B, D, horizon)  [Conv1d expects channels first]
          2. Encode timestep + obs into global conditioning vector
          3. Encoder: downsample at each level, save skip connections
          4. Bottleneck: process at deepest level
          5. Decoder: upsample at each level, concatenate skip connections
          6. Transpose back: (B, D, horizon) -> (B, horizon, D)
        """
        # Conv1d expects (B, channels, length)
        sample = einops.rearrange(sample, 'b h t -> b t h')

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        # Build global conditioning: [timestep_embed, obs_features]
        global_feature = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        h_local = []
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
            resnet, resnet2 = self.local_cond_encoder
            h_local.append(resnet(local_cond, global_feature))
            h_local.append(resnet2(local_cond, global_feature))

        # ---- Encoder (downsampling path) ----
        x = sample
        h = []  # skip connections
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)  # save skip
            x = downsample(x)

        # ---- Bottleneck ----
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # ---- Decoder (upsampling path + skip connections) ----
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)  # channel-wise concat skip
            x = resnet(x, global_feature)
            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        x = einops.rearrange(x, 'b t h -> b h t')
        return x


if __name__ == '__main__':
    emb = SinusoidalPosEmb(dim=32)
    t = torch.tensor([0,50,99])
    out = emb(t)
    print(f"shape: {out.shape}")
    print(f"t=0 != t=50: {not torch.allclose(out[0], out[1])}")
    print(f"t=50 != t=99: {not torch.allclose(out[1], out[2])}")

    block = Conv1dBlock(in_channels=16, out_channels=32, kernel_size=3)
    x = torch.randn(4, 16, 100)
    out = block(x)
    print(f"shape:{out.shape}")

    down = Downsample1d(dim=32)
    up = Upsample1d(dim=32)
    x = torch.randn(4, 32, 16)
    x_down = down(x)
    x_up = up(x_down)
    print(f"original: {x.shape}")
    print(f"after_down: {x_down.shape}")
    print(f"after_up: {x_up.shape}")

    print("\n--- ConditionalResidualBlock1D ---")
    cres = ConditionalResidualBlock1D(in_channels=16, out_channels=32, cond_dim=64)
    x = torch.randn(4, 16, 100)
    cond = torch.randn(4, 64)
    out = cres(x, cond)
    print(f"bias-only: {out.shape}")
    assert out.shape == (4, 32, 100)

    cres_scale = ConditionalResidualBlock1D(
        in_channels=32, out_channels=32, cond_dim=64, cond_predict_scale=True)
    x2 = torch.randn(4, 32, 100)
    out2 = cres_scale(x2, cond)
    print(f"scale+bias: {out2.shape}")
    assert out2.shape == (4, 32, 100)
    print("ConditionalResidualBlock1D OK")

    print("\n--- ConditionalUnet1D ---")
    B, H, action_dim = 4, 16, 2
    obs_dim = 64

    unet = ConditionalUnet1D(input_dim=action_dim, global_cond_dim=obs_dim,
                              down_dims=[64, 128, 256])
    noisy_action = torch.randn(B, H, action_dim)
    t = torch.randint(0, 100, (B,))
    obs_feat = torch.randn(B, obs_dim)
    pred = unet(noisy_action, t, global_cond=obs_feat)
    print(f"output: {pred.shape}")
    assert pred.shape == (B, H, action_dim)

    unet2 = ConditionalUnet1D(input_dim=action_dim, global_cond_dim=obs_dim,
                               local_cond_dim=3, down_dims=[64, 128, 256])
    local_c = torch.randn(B, H, 3)
    pred2 = unet2(noisy_action, t, local_cond=local_c, global_cond=obs_feat)
    print(f"with local_cond: {pred2.shape}")
    assert pred2.shape == (B, H, action_dim)

    n_params = sum(p.numel() for p in unet.parameters())
    print(f"params (global only): {n_params:,}")
    print("ConditionalUnet1D OK")
