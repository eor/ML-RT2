"""
Conditional Neural Process (theme E, stretch) -- a cheap route to per-radius uncertainty.

The parameters are encoded to a global representation; a decoder queries the (continuous)
radial coordinate and outputs a Gaussian per channel, (mean, std)(r). Trained by Gaussian
negative log-likelihood, so the model learns *where* it is uncertain (typically the sharp
front). forward() returns the mean; predict() also returns the std for error bars. A
lightweight, distinct alternative to the flow-matching uncertainty route.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .deeponet import FourierFeatures


class CNP(nn.Module):
    def __init__(self, n_params: int, n_channels: int, profile_len: int,
                 latent: int = 128, width: int = 256, n_freqs: int = 64, ff_sigma: float = 10.0):
        super().__init__()
        self.C, self.L = n_channels, profile_len
        self.repr = nn.Sequential(nn.Linear(n_params, width), nn.GELU(), nn.Linear(width, latent))
        self.ff = FourierFeatures(n_freqs, ff_sigma)
        self.dec = nn.Sequential(
            nn.Linear(latent + self.ff.out_dim(), width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, 2 * n_channels))                   # mean + log-std per channel
        self.register_buffer("coords", torch.linspace(0, 1, profile_len).view(profile_len, 1))

    def _mean_logstd(self, params):
        B = params.shape[0]
        rep = self.repr(params)[:, None, :].expand(B, self.L, -1)      # (B, L, latent)
        feat = self.ff(self.coords)[None].expand(B, -1, -1)           # (B, L, ff)
        out = self.dec(torch.cat([rep, feat], dim=-1)).view(B, self.L, self.C, 2)
        mean = out[..., 0].permute(0, 2, 1)                           # (B, C, L)
        logstd = out[..., 1].permute(0, 2, 1)
        return mean, logstd

    def forward(self, params):
        return self._mean_logstd(params)[0]

    @torch.no_grad()
    def predict(self, params):
        mean, logstd = self._mean_logstd(params)
        return mean, F.softplus(logstd) + 1e-3

    def training_loss(self, params, target):
        mean, logstd = self._mean_logstd(params)
        std = F.softplus(logstd) + 1e-3
        nll = 0.5 * (((target - mean) / std) ** 2 + 2 * torch.log(std))
        return nll.mean()


def build(cfg):
    return CNP(cfg.n_params, cfg.n_channels, cfg.profile_len,
               latent=cfg.latent, width=cfg.width, n_freqs=cfg.n_freqs, ff_sigma=cfg.ff_sigma)
