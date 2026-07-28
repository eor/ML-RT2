"""
Conditional flow matching (rectified flow) for probabilistic profile emulation.

Idea (Lipman et al. 2023; Liu et al. 2023): learn a velocity field v_theta(x, t, c)
that transports Gaussian noise x0 ~ N(0, I) at t=0 to a data profile x1 at t=1 along
the straight path
        x_t = (1 - t) x0 + t x1,   target velocity  u = x1 - x0.
Training regresses v_theta onto u (a simple MSE) -- no adversary (unlike the paper-2
CGAN) and no long diffusion chain: sampling integrates dx/dt = v_theta with a handful
of Euler steps, so inference stays cheap. Drawing several noise seeds yields an
*ensemble* -> calibrated uncertainty, which is the scientific value-add over the
deterministic operators.

Conditioning c = the 8 physical parameters. The velocity field is a dilated,
FiLM-conditioned 1D CNN (large receptive field for the long-range front dependence),
kept self-contained (torch only) for the heterogeneous cluster.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTime(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):                                   # t: (B,) in [0,1]
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


class FiLMResBlock(nn.Module):
    def __init__(self, width: int, cond_dim: int, dilation: int):
        super().__init__()
        self.conv1 = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(width, width, 3, padding=1)
        self.film = nn.Linear(cond_dim, 2 * width)
        self.norm = nn.GroupNorm(8, width)

    def forward(self, x, cond):
        h = self.conv1(x)
        scale, shift = self.film(cond).chunk(2, dim=-1)
        h = self.norm(h) * (1 + scale[..., None]) + shift[..., None]
        h = F.gelu(h)
        h = self.conv2(h)
        return x + h


class VelocityField(nn.Module):
    def __init__(self, n_channels: int, n_params: int,
                 width: int = 96, depth: int = 10, cond_dim: int = 128, time_dim: int = 64):
        super().__init__()
        self.time = SinusoidalTime(time_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(time_dim + n_params, cond_dim), nn.GELU(),
            nn.Linear(cond_dim, cond_dim), nn.GELU(),
        )
        self.lift = nn.Conv1d(n_channels, width, 1)
        self.blocks = nn.ModuleList([
            FiLMResBlock(width, cond_dim, dilation=2 ** (i % 6)) for i in range(depth)
        ])
        self.head = nn.Conv1d(width, n_channels, 1)

    def forward(self, x, t, params):                        # x:(B,C,L) t:(B,) params:(B,P)
        cond = self.cond_mlp(torch.cat([self.time(t), params], dim=-1))
        h = self.lift(x)
        for b in self.blocks:
            h = b(h, cond)
        return self.head(h)


class FlowMatching(nn.Module):
    def __init__(self, n_params: int, n_channels: int, profile_len: int,
                 width: int = 96, depth: int = 10, eval_steps: int = 20):
        super().__init__()
        self.C, self.L = n_channels, profile_len
        self.net = VelocityField(n_channels, n_params, width=width, depth=depth)
        self.eval_steps = eval_steps

    # --- training objective (rectified-flow / conditional OT path) ---
    def training_loss(self, params, x1):
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=x1.device)
        xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * x1
        v = self.net(xt, t, params)
        return F.mse_loss(v, x1 - x0)

    # --- sampling: integrate dx/dt = v from noise (t=0) to data (t=1) ---
    @torch.no_grad()
    def sample(self, params, steps: int | None = None, n: int = 1):
        steps = steps or self.eval_steps
        B = params.shape[0]
        dev = params.device
        if n > 1:
            params = params.repeat_interleave(n, dim=0)
        x = torch.randn(params.shape[0], self.C, self.L, device=dev)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((params.shape[0],), i * dt, device=dev)
            x = x + self.net(x, t, params) * dt
        return x.view(B, n, self.C, self.L) if n > 1 else x

    # point estimate so the generic evaluate()/analyze() work unchanged
    def forward(self, params):
        return self.sample(params, steps=self.eval_steps, n=1)


def build(cfg):
    return FlowMatching(cfg.n_params, cfg.n_channels, cfg.profile_len,
                        width=cfg.width, depth=cfg.depth, eval_steps=cfg.eval_steps)
