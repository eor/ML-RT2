"""
DeepONet for parameter -> multi-field profile emulation.

  branch(params) -> per-channel coefficient vectors b_{c,k}
  trunk(r)       -> shared basis functions t_k(r)
  profile_c(r)   = sum_k b_{c,k} t_k(r) + bias_c

The trunk uses random Fourier features of the (continuous) radial coordinate, which
lets a small network represent the sharp ionization front. Because the trunk is a
function of r, inference is continuous in radius and extremely cheap (a matmul), and
the emulator is resolution-free -- both attractive for the 'capable + fast inference'
goal. Novel-for-astro as an operator-learning emulator, and a clean comparison point
against the FNO.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    def __init__(self, n_freqs: int = 64, sigma: float = 10.0):
        super().__init__()
        self.register_buffer("B", torch.randn(1, n_freqs) * sigma)

    def out_dim(self):
        return 2 * self.B.shape[1]

    def forward(self, r):                                   # r: (L, 1)
        proj = 2 * math.pi * r @ self.B                     # (L, n_freqs)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


def _mlp(i, o, h, n):
    layers = [nn.Linear(i, h), nn.GELU()]
    for _ in range(n - 1):
        layers += [nn.Linear(h, h), nn.GELU()]
    layers += [nn.Linear(h, o)]
    return nn.Sequential(*layers)


class DeepONet(nn.Module):
    def __init__(self, n_params: int, n_channels: int, profile_len: int,
                 latent: int = 128, width: int = 256, depth: int = 4,
                 n_freqs: int = 64, ff_sigma: float = 10.0):
        super().__init__()
        self.C, self.K, self.L = n_channels, latent, profile_len
        self.ff = FourierFeatures(n_freqs, ff_sigma)
        self.branch = _mlp(n_params, n_channels * latent, width, depth)
        self.trunk = _mlp(self.ff.out_dim(), latent, width, depth)
        self.bias = nn.Parameter(torch.zeros(n_channels))
        self.register_buffer("coords", torch.linspace(0, 1, profile_len).view(profile_len, 1))

    def forward(self, params, coords=None):                 # params: (B, n_params)
        B = params.shape[0]
        r = self.coords if coords is None else coords
        b = self.branch(params).view(B, self.C, self.K)     # (B, C, K)
        t = self.trunk(self.ff(r))                           # (L, K)
        out = torch.einsum("bck,lk->bcl", b, t) + self.bias.view(1, self.C, 1)
        return out                                           # (B, C, L)


def build(cfg):
    return DeepONet(cfg.n_params, cfg.n_channels, cfg.profile_len,
                    latent=cfg.latent, width=cfg.width, depth=cfg.depth,
                    n_freqs=cfg.n_freqs, ff_sigma=cfg.ff_sigma)
