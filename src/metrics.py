"""
Metrics for the paper-3 emulators.

Two spaces:
  * encoded space  -> the standardized-log space the models train in (primary loss)
  * physical space -> after Normalizer.decode_profiles (fractions in [0,1], T in K)

We keep a set that supports the cross-architecture 'how do models optimise' study:
per-channel errors, a physical relative-L2, and an ionization-front-position error.
"""
from __future__ import annotations
import numpy as np
import torch
from data import CHANNELS, PROFILE_LEN


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def per_channel_mse(pred: torch.Tensor, target: torch.Tensor) -> dict:
    # pred/target: (B, C, L)
    m = torch.mean((pred - target) ** 2, dim=(0, 2))
    return {CHANNELS[c]: float(m[c]) for c in range(len(CHANNELS))}


def relative_l2_physical(pred_phys: np.ndarray, target_phys: np.ndarray) -> dict:
    """Mean per-sample relative L2 error in physical units, per channel."""
    num = np.sqrt(((pred_phys - target_phys) ** 2).sum(axis=2))
    den = np.sqrt((target_phys ** 2).sum(axis=2)) + 1e-12
    rel = (num / den).mean(axis=0)                          # (C,)
    return {CHANNELS[c]: float(rel[c]) for c in range(len(CHANNELS))}


def front_position_error(pred_phys: np.ndarray, target_phys: np.ndarray,
                         channel: int = 0, threshold: float = 0.5) -> float:
    """
    Ionization-front position error (in grid cells) on x_HII.

    The front is where x_HII drops through `threshold`. We take the first crossing
    from the source outward and report the mean absolute index difference. This is a
    physically meaningful, sharp-feature metric that plain MSE can hide.
    """
    def first_crossing(arr):                                # arr: (B, L)
        below = arr < threshold
        idx = np.argmax(below, axis=1).astype(np.float64)   # first True
        idx[~below.any(axis=1)] = PROFILE_LEN - 1           # never crosses -> last cell
        return idx
    fp = first_crossing(pred_phys[:, channel])
    ft = first_crossing(target_phys[:, channel])
    return float(np.mean(np.abs(fp - ft)))


@torch.no_grad()
def evaluate(model_fn, loader, device, normalizer=None) -> dict:
    """
    Aggregate metrics over a loader. `model_fn(params) -> (B, C, L)` in encoded space.
    Returns encoded MSE (overall + per channel) and, if a normalizer is given,
    physical relative-L2 and front-position error.
    """
    sq_err = 0.0
    n = 0
    per_c = np.zeros(len(CHANNELS))
    preds_phys, targs_phys = [], []
    for params, target in loader:
        params = params.to(device)
        pred = model_fn(params).float().cpu()
        sq_err += float(((pred - target) ** 2).sum())
        per_c += ((pred - target) ** 2).sum(dim=(0, 2)).numpy()
        n += target.numel()
        if normalizer is not None:
            preds_phys.append(normalizer.decode_profiles(pred))
            targs_phys.append(normalizer.decode_profiles(target))
    out = {"mse": sq_err / n}
    per_c /= (len(loader.dataset) * PROFILE_LEN)
    out.update({f"mse_{CHANNELS[c]}": float(per_c[c]) for c in range(len(CHANNELS))})
    if normalizer is not None:
        pp = np.concatenate(preds_phys, 0)
        tt = np.concatenate(targs_phys, 0)
        out["rel_l2"] = relative_l2_physical(pp, tt)
        out["front_err_cells"] = front_position_error(pp, tt)
    return out
