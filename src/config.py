"""Config for the paper-3 emulators (dataclass-based, CLI-overridable)."""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
import json


@dataclass
class Config:
    # --- data ---
    data_dir: str = "../data/053_data_set"
    n_params: int = 8
    profile_len: int = 1500
    n_channels: int = 4
    subset: int | None = None            # use a random subset for quick local runs; None = full
    standardize: bool = True
    batch_size: int = 256

    # --- model (generic; individual models read what they need) ---
    model: str = "fno"                   # fno | deeponet | ...
    width: int = 64                      # channel width / hidden size
    depth: int = 4                       # number of blocks / layers
    modes: int = 32                      # FNO: retained Fourier modes
    latent: int = 128                    # DeepONet: branch/trunk latent dim
    n_freqs: int = 64                    # DeepONet: number of Fourier features on r
    ff_sigma: float = 10.0               # DeepONet: Fourier-feature bandwidth (front sharpness)
    eval_steps: int = 20                 # flow matching: ODE steps for the point estimate
    physics_weight: float = 0.1          # PINO: weight of the physics residual term
    pino_backbone: str = "fno"           # PINO: operator backbone (fno | deeponet)
    node_steps: int = 128                # Neural-ODE decoder: integration steps in radius

    # --- optimisation ---
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_accum: int = 1                  # effective batch = batch_size * grad_accum
    amp: bool = True                     # bf16 autocast (works 3090 -> H200)
    amp_dtype: str = "bfloat16"          # bfloat16 | float16 | float32
    early_stop_patience: int = 30
    clip_grad: float = 1.0

    # --- run bookkeeping ---
    out_dir: str = "output"
    run_name: str = "run"
    seed: int = 42
    device: str = "auto"                 # auto | cpu | cuda
    ckpt_every: int = 10                 # save a resumable checkpoint every N epochs
    resume: str | None = None            # path to checkpoint to resume from
    log_every: int = 50                  # steps

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_args(cls, argv=None):
        """Minimal CLI: --key value / --flag (bools). Optional (None-default) fields are
        typed from their annotation (e.g. `subset: int | None` parses as int)."""
        import argparse, dataclasses

        def optional_conv(type_str):
            def conv(s):
                if s is None or str(s).lower() in ("none", ""):
                    return None
                if "int" in type_str:
                    return int(s)
                if "float" in type_str:
                    return float(s)
                return s
            return conv

        p = argparse.ArgumentParser()
        defaults = asdict(cls())
        for fld in dataclasses.fields(cls):
            k, v, tstr = fld.name, defaults[fld.name], str(fld.type)
            if isinstance(v, bool):
                p.add_argument(f"--{k}", type=lambda s: s.lower() in ("1", "true", "yes"), default=v)
            elif "None" in tstr:                       # Optional[...] -> type from annotation
                p.add_argument(f"--{k}", type=optional_conv(tstr), default=v)
            else:
                p.add_argument(f"--{k}", type=type(v), default=v)
        return cls(**vars(p.parse_args(argv)))
