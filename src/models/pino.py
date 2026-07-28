"""
Physics-informed neural operator (PINO).

  * a backbone operator (FNO by default) predicts the profile in encoded space;
  * a small 'physics head' maps the 8 parameters to per-sample amplitudes
    (Gamma0, kappa) used by the grey ionisation-equilibrium residual;
  * training loss = data MSE + physics_weight * (equilibrium + boundedness + monotonicity),
    all evaluated on a differentiable decode of the prediction into physical units.

This is the data-anchored route: the operator already fits the STARDUST solutions, and the
soft physics term refines them toward ionisation-equilibrium consistency (see physics.py for
the rationale and why we do NOT reuse the old time-evolution PINN residual).

forward(params) returns the encoded profile, so the generic evaluate()/analyze() work unchanged.
"""
from __future__ import annotations
import torch
import torch.nn as nn

import physics as phys
from . import fno, deeponet


class PINO(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = (deeponet.build(cfg) if cfg.pino_backbone == "deeponet"
                         else fno.build(cfg))
        # physics head: params -> (log Gamma0, log kappa); softplus -> positive amplitudes
        self.phys_head = nn.Sequential(
            nn.Linear(cfg.n_params, 32), nn.GELU(), nn.Linear(32, 2))
        self.physics_weight = cfg.physics_weight
        self.register_buffer("r_grid", torch.linspace(0, 1, cfg.profile_len).clamp_min(1e-3))
        # decode stats (overwritten by set_normalizer); defaults => identity standardize
        C = cfg.n_channels
        self.register_buffer("chan_mean", torch.zeros(C))
        self.register_buffer("chan_std", torch.ones(C))
        self.register_buffer("is_fraction", torch.tensor([1., 1., 1., 0.]))
        self.log_floor = 1e-6

    # trainer calls this so the physics term decodes to correct physical units
    def set_normalizer(self, norm):
        if norm is None or norm.chan_mean is None:
            return
        self.chan_mean = torch.tensor(norm.chan_mean, dtype=torch.float32)
        self.chan_std = torch.tensor(norm.chan_std, dtype=torch.float32)
        frac = [1.0 if f else 0.0 for f in norm.channel_is_fraction]
        self.is_fraction = torch.tensor(frac, dtype=torch.float32)

    def _decode(self, enc):                                    # enc: (B, C, L) -> physical channels
        x = enc * self.chan_std[None, :, None] + self.chan_mean[None, :, None]
        x = x.clamp(-20.0, 12.0)                               # guard 10**x overflow
        p = torch.pow(10.0, x) - self.is_fraction[None, :, None] * self.log_floor
        return p[:, 0], p[:, 1], p[:, 2], p[:, 3]              # xHII, xHeII, xHeIII, T

    def forward(self, params):
        return self.backbone(params)

    def physics_loss(self, pred_enc, params):
        xHII, xHeII, xHeIII, T = self._decode(pred_enc)
        T = torch.clamp(T, min=1.0)
        amp = torch.nn.functional.softplus(self.phys_head(params))   # (B, 2) > 0
        gamma0, kappa = amp[:, 0] + 1e-6, amp[:, 1] + 1e-6
        return (phys.hydrogen_equilibrium_residual(xHII, T, gamma0, kappa, self.r_grid)
                + phys.boundedness_penalty(xHII, xHeII, xHeIII, T)
                + phys.monotonicity_penalty(xHII, xHeIII))

    def training_loss(self, params, target):
        pred = self.backbone(params)
        data = torch.mean((pred - target) ** 2)
        phys_term = self.physics_loss(pred, params)
        return data + self.physics_weight * phys_term


def build(cfg):
    return PINO(cfg)
