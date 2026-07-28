"""
Dataset loader for the paper-3 emulators.

The emulation task: map 8 physical parameters -> 4 radial profiles
(x_HII, x_HeII, x_HeIII, T), each PROFILE_LEN points over radius.

Transforms are chosen to match ML-RT papers 1-2 for comparability:
  * parameters       : min-max scaled to [0, 1] using P8_LIMITS
  * fraction profiles: log10(x + 1e-6)   (x in [0,1] -> log in [-6, 0])
  * temperature      : log10(T)
On top of the log transform we optionally z-score each channel using
*training-set* statistics (recommended for stable training; the stats are
stored on the Normalizer so predictions can be inverted back to physical units).

Nothing here imports from the (read-only) ML-RT / ML-RT2 code; the relevant
conventions are duplicated so this subtree is self-contained.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

PROFILE_LEN = 1500
CHANNELS = ("HII", "HeII", "HeIII", "T")          # output profile channels, in order
PROFILE_FILES = {
    "HII":   "data_HII_profiles.npy",
    "HeII":  "data_HeII_profiles.npy",
    "HeIII": "data_HeIII_profiles.npy",
    "T":     "data_T_profiles.npy",
}
PARAM_FILE = "data_parameters.npy"

# 8-parameter limits (from ML-RT settings_parameters.py) -- keep in sync.
P8_LIMITS = np.array([
    [8.0, 15.0],                                    # log10 halo mass
    [6.0, 13.0],                                    # redshift
    [0.1, 20.0],                                    # source age [Myr]
    [0.0, 2.0],                                     # QSO alpha
    [0.0, 1.0],                                     # QSO efficiency
    [0.0, 1.0],                                     # stellar escape fraction
    [0.0, 2.5],                                     # IMF slope
    [0.6989700043360189, 2.6989700043360187],       # log10 IMF min mass (log10[5,500])
], dtype=np.float64)

LOG_FLOOR = 1.0e-6                                   # matches papers 1-2


@dataclass
class Normalizer:
    """Holds the transforms so predictions can be inverted to physical units."""
    param_limits: np.ndarray
    channel_is_fraction: tuple = (True, True, True, False)   # HII/HeII/HeIII fraction, T not
    standardize: bool = True
    chan_mean: np.ndarray | None = None             # per-channel mean of log profiles (train set)
    chan_std: np.ndarray | None = None

    # --- parameters ---
    def scale_params(self, p: np.ndarray) -> np.ndarray:
        a, b = self.param_limits[:, 0], self.param_limits[:, 1]
        return (p - a) / (b - a)

    def unscale_params(self, p: np.ndarray) -> np.ndarray:
        a, b = self.param_limits[:, 0], self.param_limits[:, 1]
        return p * (b - a) + a

    # --- profiles ---  (input shape: (N, 4, PROFILE_LEN), physical units)
    def encode_profiles(self, prof: np.ndarray) -> np.ndarray:
        out = np.empty_like(prof, dtype=np.float32)
        for c, is_frac in enumerate(self.channel_is_fraction):
            out[:, c] = np.log10(prof[:, c] + (LOG_FLOOR if is_frac else 0.0))
        if self.standardize and self.chan_mean is not None:
            out = (out - self.chan_mean[None, :, None]) / self.chan_std[None, :, None]
        return out

    def decode_profiles(self, enc: np.ndarray | torch.Tensor) -> np.ndarray:
        x = enc.detach().cpu().numpy() if isinstance(enc, torch.Tensor) else np.asarray(enc)
        x = x.astype(np.float64)
        if self.standardize and self.chan_mean is not None:
            x = x * self.chan_std[None, :, None] + self.chan_mean[None, :, None]
        out = np.empty_like(x)
        for c, is_frac in enumerate(self.channel_is_fraction):
            out[:, c] = np.power(10.0, x[:, c]) - (LOG_FLOOR if is_frac else 0.0)
        return out


class ProfileDataset(Dataset):
    """In-memory (or subset) params -> (4, PROFILE_LEN) profiles."""
    def __init__(self, params: np.ndarray, profiles: np.ndarray):
        assert params.shape[0] == profiles.shape[0]
        self.params = torch.from_numpy(params.astype(np.float32))
        self.profiles = torch.from_numpy(profiles.astype(np.float32))

    def __len__(self):
        return self.params.shape[0]

    def __getitem__(self, i):
        return self.params[i], self.profiles[i]


def _load_raw(data_dir: str, subset: int | None, seed: int):
    """Load params + the 4 profile arrays (optionally a random subset for quick tests)."""
    params = np.load(os.path.join(data_dir, PARAM_FILE), mmap_mode="r")
    n_total = params.shape[0]
    if subset is not None and subset < n_total:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n_total, size=subset, replace=False))
    else:
        idx = slice(None)
    params = np.asarray(params[idx], dtype=np.float64)
    profs = []
    for ch in CHANNELS:
        arr = np.load(os.path.join(data_dir, PROFILE_FILES[ch]), mmap_mode="r")
        profs.append(np.asarray(arr[idx], dtype=np.float64))
    profiles = np.stack(profs, axis=1)              # (N, 4, PROFILE_LEN)
    return params, profiles


def load_dataset(
    data_dir: str,
    subset: int | None = None,
    split=(0.80, 0.10, 0.10),
    seed: int = 42,
    standardize: bool = True,
):
    """Return (train_ds, val_ds, test_ds, normalizer).

    subset: if given, use a random subset of this many samples (for fast local sanity).
    Split fractions and the shuffle seed default to the paper 1-2 values so the
    test partition lines up for direct comparison when subset is None.
    """
    params, profiles = _load_raw(data_dir, subset, seed)
    n = params.shape[0]

    # deterministic shuffle + split (matches SHUFFLE_SEED=42 convention)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_tr = int(split[0] * n)
    n_va = int(split[1] * n)
    tr, va, te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]

    norm = Normalizer(param_limits=P8_LIMITS, standardize=standardize)
    p_scaled = norm.scale_params(params).astype(np.float32)

    # log-transform first, then compute standardization stats on the TRAIN split only
    log_prof = np.empty_like(profiles, dtype=np.float32)
    for c, is_frac in enumerate(norm.channel_is_fraction):
        log_prof[:, c] = np.log10(profiles[:, c] + (LOG_FLOOR if is_frac else 0.0))
    if standardize:
        norm.chan_mean = log_prof[tr].mean(axis=(0, 2)).astype(np.float32)          # (4,)
        norm.chan_std = (log_prof[tr].std(axis=(0, 2)) + 1e-8).astype(np.float32)
        log_prof = (log_prof - norm.chan_mean[None, :, None]) / norm.chan_std[None, :, None]

    make = lambda ix: ProfileDataset(p_scaled[ix], log_prof[ix])
    return make(tr), make(va), make(te), norm


def make_loaders(train_ds, val_ds, test_ds, batch_size=256, num_workers=0):
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return (
        DataLoader(train_ds, shuffle=True, drop_last=True, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
        DataLoader(test_ds, shuffle=False, **kw),
    )


def synthetic_dataset(n=512, n_params=8, seed=0):
    """A tiny, structured synthetic task for shape/overfit sanity checks without the real data.

    Builds smooth radial profiles with a parameter-dependent 'ionization front' so that a
    working model can overfit a batch. Values are already in the encoded (standardized-ish)
    space, so no Normalizer is needed.
    """
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, 1, size=(n, n_params)).astype(np.float32)
    r = np.linspace(0, 1, PROFILE_LEN, dtype=np.float32)[None, :]
    front = (0.2 + 0.6 * p[:, 0:1])                 # front radius from param 0
    width = (0.02 + 0.1 * p[:, 1:2])
    hii = 0.5 * (1 - np.tanh((r - front) / width))  # ionized -> neutral transition
    heii = 0.4 * np.exp(-((r - front) ** 2) / (2 * (width * 2) ** 2))
    heiii = 0.5 * (1 - np.tanh((r - 0.5 * front) / width))
    temp = 1.0 - 0.8 * r + 0.3 * p[:, 2:3] * np.cos(6 * r)
    prof = np.stack([hii, heii, heiii, temp], axis=1).astype(np.float32)
    return ProfileDataset(p, prof)


if __name__ == "__main__":
    ds = synthetic_dataset(16)
    x, y = ds[0]
    print("synthetic sample:", x.shape, y.shape)      # (8,), (4, 1500)
