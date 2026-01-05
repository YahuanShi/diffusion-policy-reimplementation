import torch
import torch.nn as nn
from types import SimpleNamespace

class SinusoidalPosEmb(nn.Module):
    """Encode a scalar timestep t into a vector of dimension 'dim'.
        Standard in transformers and diffusion models
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        freqs = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(half_dim) / half_dim)
        emb = t.unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


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
    def __init__(self, in_channels, out_channels, cond_dim,
                 kernel_size=3, n_groups=8, cond_predict_scale=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation: predict per-channel bias (and optionally scale)
        cond_channels = out_channels * 2 if cond_predict_scale else out_channels
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
        )

        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        out = self.blocks[0](x)
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
