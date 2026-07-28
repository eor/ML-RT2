"""
Rudimentary, reusable plotting for the paper-3 emulators.

  * plot_sample_profiles : emulator vs STARDUST for a few test samples, physical units
  * plot_history         : training dynamics (loss curves, per-channel val, grad norm,
                           epoch time) from one or more history.json runs -- the backbone
                           of the cross-architecture 'how do models optimise' comparison

Headless (Agg) so it runs on cluster nodes; every function saves to a file.
"""
from __future__ import annotations
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import CHANNELS, PROFILE_LEN

_R = np.linspace(0, 1, PROFILE_LEN)     # normalized radius (relabel to kpc later if wanted)


def plot_sample_profiles(preds_phys, targs_phys, indices, out_path):
    """preds/targs: (N, 4, L) in physical units."""
    n = len(indices)
    fig, axes = plt.subplots(n, 4, figsize=(15, 2.6 * n), squeeze=False)
    for i, s in enumerate(indices):
        for c in range(4):
            ax = axes[i][c]
            ax.plot(_R, targs_phys[s, c], "k-", lw=1.6, label="STARDUST")
            ax.plot(_R, preds_phys[s, c], "r--", lw=1.2, label="emulator")
            if CHANNELS[c] == "T":
                ax.set_yscale("log")
            else:
                ax.set_ylim(-0.05, 1.05)
            if i == 0:
                ax.set_title(CHANNELS[c])
            if i == 0 and c == 0:
                ax.legend(fontsize=8, loc="best")
            if i == n - 1:
                ax.set_xlabel("r (normalized)")
    fig.suptitle("emulator vs STARDUST (test samples)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _load_history(h):
    if isinstance(h, str):
        with open(h) as f:
            return json.load(f)
    return h


def plot_history(histories, out_path, labels=None):
    """histories: a history dict/path, or a list of them (to overlay architectures)."""
    if not isinstance(histories, (list, tuple)):
        histories = [histories]
    hs = [_load_history(h) for h in histories]
    labels = labels or [f"run {i}" for i in range(len(hs))]

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for h, lab in zip(hs, labels):
        ep = np.arange(len(h["train_loss"]))
        val_mse = [v["mse"] for v in h["val"]]
        ax[0, 0].plot(ep, h["train_loss"], label=f"{lab} train")
        ax[0, 0].plot(ep, val_mse, "--", label=f"{lab} val")
        # per-channel val mse (final run's channels)
        for c in CHANNELS:
            key = f"mse_{c}"
            if key in h["val"][0]:
                ax[0, 1].plot(ep, [v[key] for v in h["val"]], label=f"{lab}:{c}")
        ax[1, 0].plot(ep, h["grad_norm"], label=lab)
        ax[1, 1].plot(np.cumsum(h["epoch_time"]) / 60.0, val_mse, label=lab)

    ax[0, 0].set(title="loss (train solid / val dashed)", xlabel="epoch", ylabel="MSE"); ax[0, 0].set_yscale("log"); ax[0, 0].legend(fontsize=7)
    ax[0, 1].set(title="per-channel val MSE", xlabel="epoch"); ax[0, 1].set_yscale("log"); ax[0, 1].legend(fontsize=7)
    ax[1, 0].set(title="gradient norm", xlabel="epoch"); ax[1, 0].legend(fontsize=7)
    ax[1, 1].set(title="val MSE vs wall-clock", xlabel="minutes", ylabel="val MSE"); ax[1, 1].set_yscale("log"); ax[1, 1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
