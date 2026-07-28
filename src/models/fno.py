"""
1D Fourier Neural Operator for parameter -> multi-field profile emulation.

Why FNO here (novelty + fit): the outputs are smooth 1D fields over radius with a
sharp ionization front; spectral convolutions capture multiscale structure and the
operator is resolution-invariant (train at 1500, query any grid). The input is a
small parameter vector, so we lift it to a field: every radial location receives the
(same) parameters plus its own normalized r-coordinate, then FNO blocks act along r.

Self-contained (only torch) so it drops onto any cluster node without extra packages.
For the production runs we can optionally swap in the `neuraloperator` library.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """Multiply the lowest `modes` Fourier modes by a learnable complex matrix."""
    def __init__(self, in_ch: int, out_ch: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (in_ch * out_ch)
        self.weight = nn.Parameter(scale * torch.rand(in_ch, out_ch, modes, dtype=torch.cfloat))

    def forward(self, x):                                   # x: (B, C, L)
        B, C, L = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)                    # (B, C, L//2+1)
        m = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(B, self.weight.shape[1], x_ft.shape[-1],
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :m] = torch.einsum("bim,iom->bom", x_ft[:, :, :m], self.weight[:, :, :m])
        return torch.fft.irfft(out_ft, n=L, dim=-1)         # (B, out_ch, L)


class FNO1d(nn.Module):
    def __init__(self, n_params: int, n_channels: int, profile_len: int,
                 width: int = 64, modes: int = 32, depth: int = 4):
        super().__init__()
        self.L = profile_len
        self.lift = nn.Linear(n_params + 1, width)          # +1 = normalized r-coordinate
        self.spectral = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(depth)])
        self.pointwise = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(depth)])
        self.proj = nn.Sequential(
            nn.Conv1d(width, 2 * width, 1), nn.GELU(),
            nn.Conv1d(2 * width, n_channels, 1),
        )
        self.register_buffer("grid", torch.linspace(0, 1, profile_len).view(1, 1, profile_len))

    def forward(self, params):                              # params: (B, n_params)
        B, P = params.shape
        grid = self.grid.expand(B, 1, self.L)
        pfield = params.unsqueeze(-1).expand(B, P, self.L)  # broadcast params over r
        x = torch.cat([pfield, grid], dim=1)                # (B, P+1, L)
        x = self.lift(x.transpose(1, 2)).transpose(1, 2)    # (B, width, L)
        for sp, pw in zip(self.spectral, self.pointwise):
            x = x + F.gelu(sp(x) + pw(x))                   # residual FNO block
        return self.proj(x)                                 # (B, n_channels, L)


def build(cfg):
    return FNO1d(cfg.n_params, cfg.n_channels, cfg.profile_len,
                 width=cfg.width, modes=cfg.modes, depth=cfg.depth)
