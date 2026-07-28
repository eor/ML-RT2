"""
Neural-ODE decoder in radius (theme E, stretch).

A physically-structured 'marching' emulator: a latent state h(r) is initialised from the
parameters and integrated outward,
        dh/dr = f(h, r, theta),      u(r) = readout(h(r)),
mirroring how the ionisation structure builds up with radius. Implemented as an explicit
fixed-step integrator (a ResNet in r) so it is self-contained -- no torchdiffeq dependency on
the cluster. Integrates on a coarse grid and upsamples, keeping it cheap and resolution-flexible.
Deterministic -> generic MSE loss.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralODEProfile(nn.Module):
    def __init__(self, n_params: int, n_channels: int, profile_len: int,
                 hidden: int = 64, steps: int = 128):
        super().__init__()
        self.C, self.L, self.steps = n_channels, profile_len, steps
        self.init = nn.Sequential(nn.Linear(n_params, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.param_emb = nn.Linear(n_params, hidden)
        self.f = nn.Sequential(nn.Linear(hidden + 1 + hidden, hidden), nn.GELU(),
                               nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.readout = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n_channels))

    def forward(self, params):                                  # params: (B, n_params)
        B = params.shape[0]
        h = self.init(params)
        te = self.param_emb(params)
        dr = 1.0 / self.steps
        outs = []
        for i in range(self.steps):
            r = torch.full((B, 1), i * dr, device=params.device, dtype=params.dtype)
            h = h + dr * self.f(torch.cat([h, r, te], dim=-1))  # explicit Euler step
            outs.append(self.readout(h))
        u = torch.stack(outs, dim=-1)                           # (B, C, steps)
        if self.steps != self.L:
            u = F.interpolate(u, size=self.L, mode="linear", align_corners=True)
        return u


def build(cfg):
    return NeuralODEProfile(cfg.n_params, cfg.n_channels, cfg.profile_len,
                            hidden=cfg.width, steps=cfg.node_steps)
