"""
Profile / signal transformer decoder (theme D).

The 8 parameters are embedded into a conditioning token; a set of learned query tokens --
one per radial *patch* -- attend to that token and to each other, then each decodes its patch
of the 4-channel profile. Self-attention couples long-range structure (front position vs.
integrated absorption) and, through the shared tokens, the four species. A fresh architecture
for RT emulation; deterministic, so it trains with the generic MSE loss.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class ProfileTransformer(nn.Module):
    def __init__(self, n_params: int, n_channels: int, profile_len: int,
                 d_model: int = 128, depth: int = 4, nhead: int = 4, n_patches: int = 50):
        super().__init__()
        assert profile_len % n_patches == 0, "profile_len must be divisible by n_patches"
        self.C, self.L, self.NP, self.PS = n_channels, profile_len, n_patches, profile_len // n_patches
        self.param_embed = nn.Sequential(nn.Linear(n_params, d_model), nn.GELU(),
                                         nn.Linear(d_model, d_model))
        self.query = nn.Parameter(torch.randn(n_patches, d_model) * 0.02)   # radial patch tokens
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4 * d_model,
                                           batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.head = nn.Linear(d_model, n_channels * self.PS)

    def forward(self, params):                                  # params: (B, n_params)
        B = params.shape[0]
        ptok = self.param_embed(params).unsqueeze(1)            # (B, 1, d)
        q = self.query.unsqueeze(0).expand(B, -1, -1)           # (B, NP, d)
        out = self.encoder(torch.cat([ptok, q], dim=1))[:, 1:]  # (B, NP, d)  (drop param token)
        patch = self.head(out).view(B, self.NP, self.C, self.PS)
        return patch.permute(0, 2, 1, 3).reshape(B, self.C, self.L)


def build(cfg):
    nhead = 4 if cfg.width % 4 == 0 else 2
    return ProfileTransformer(cfg.n_params, cfg.n_channels, cfg.profile_len,
                              d_model=cfg.width, depth=cfg.depth, nhead=nhead)
