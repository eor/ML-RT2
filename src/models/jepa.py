"""
Joint-Embedding Predictive Architecture (theme E, stretch).

Trains in *embedding* space: a parameter encoder and a (target) profile encoder produce
latents; a predictor maps the parameter-latent to the profile-latent, and a decoder turns
that predicted latent into a profile. Loss = reconstruction + latent-prediction:
        L = || decode(pred) - u ||^2  +  beta || pred - stopgrad(enc_profile(u)) ||^2 .
The latent term encourages a predictive, abstraction-first representation (the JEPA idea);
the reconstruction anchor prevents representation collapse. Almost unused in astrophysics ->
a genuinely novel angle; also a shared latent useful for interpolation.

(Refinement for later: an EMA / momentum target encoder, as in I-JEPA.)
"""
from __future__ import annotations
import torch
import torch.nn as nn


class JEPA(nn.Module):
    def __init__(self, n_params: int, n_channels: int, profile_len: int,
                 latent: int = 128, width: int = 256, beta: float = 1.0):
        super().__init__()
        self.C, self.L, self.beta = n_channels, profile_len, beta
        self.param_enc = nn.Sequential(nn.Linear(n_params, width), nn.GELU(), nn.Linear(width, latent))
        self.target_enc = nn.Sequential(
            nn.Conv1d(n_channels, width, 7, stride=4, padding=3), nn.GELU(),
            nn.Conv1d(width, width, 5, stride=4, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(width, latent))
        self.predictor = nn.Sequential(nn.Linear(latent, latent), nn.GELU(), nn.Linear(latent, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, width), nn.GELU(),
                                     nn.Linear(width, n_channels * profile_len))

    def forward(self, params):                                  # inference: params -> profile
        z = self.predictor(self.param_enc(params))
        return self.decoder(z).view(-1, self.C, self.L)

    def training_loss(self, params, target):
        z_pred = self.predictor(self.param_enc(params))
        z_targ = self.target_enc(target).detach()               # stop-grad target embedding
        latent_loss = torch.mean((z_pred - z_targ) ** 2)
        recon = self.decoder(z_pred).view(-1, self.C, self.L)
        recon_loss = torch.mean((recon - target) ** 2)
        return recon_loss + self.beta * latent_loss


def build(cfg):
    return JEPA(cfg.n_params, cfg.n_channels, cfg.profile_len, latent=cfg.latent, width=cfg.width)
