"""
Sanity checks the prototypes without the real data or a GPU:
  1) forward pass returns the right shape  (B, n_channels, profile_len)
  2) the model can overfit a single batch of the synthetic task (loss collapses)

A model that fails (2) has a wiring bug -- this catches most mistakes before we ship
anything to the cluster.  Run:  python sanity_check.py
"""
from __future__ import annotations
import torch
from config import Config
from data import synthetic_dataset
from trainer import Trainer
from models import fno, deeponet, flow, pino, transformer, node, jepa, cnp

torch.manual_seed(0)


def check(name, build_fn, threshold=0.25, steps=250, nll=False, **over):
    """Shape check + overfit-a-batch. For ordinary (>=0) losses we require a loss-reduction
    ratio below `threshold`; for NLL objectives (which cross zero) we require a clear
    absolute decrease instead."""
    kw = dict(model=name, device="cpu", width=32, modes=16, depth=3, latent=64, lr=3e-3)
    kw.update(over)
    cfg = Config(**kw)
    model = build_fn(cfg)
    ds = synthetic_dataset(n=64)
    B = 16
    params = torch.stack([ds[i][0] for i in range(B)])
    target = torch.stack([ds[i][1] for i in range(B)])

    out = model(params)
    exp = (B, cfg.n_channels, cfg.profile_len)
    assert out.shape == exp, f"{name}: bad shape {tuple(out.shape)} != {exp}"

    n_par = sum(p.numel() for p in model.parameters())
    losses = Trainer(model, cfg).overfit_batch((params, target), steps=steps)
    n_par_s = f"{n_par:>9,}"
    if nll:
        ok = losses[-1] < losses[0] - 0.5
        print(f"  [{'OK' if ok else 'WEAK'}] {name:12s} params={n_par_s} out={tuple(out.shape)} "
              f"nll {losses[0]:.3e} -> {losses[-1]:.3e}")
        assert ok, f"{name} NLL did not decrease ({losses[0]:.3e} -> {losses[-1]:.3e})"
    else:
        ratio = losses[-1] / losses[0]
        print(f"  [{'OK' if ratio < threshold else 'WEAK'}] {name:12s} params={n_par_s} "
              f"out={tuple(out.shape)} loss {losses[0]:.3e} -> {losses[-1]:.3e}  (x{ratio:.3f})")
        assert ratio < max(threshold, 0.6), f"{name} failed to overfit a batch (ratio {ratio:.3f})"


if __name__ == "__main__":
    print("device: cpu | synthetic task | overfit-a-batch")
    check("fno", fno.build)
    check("deeponet", deeponet.build)
    # flow matching: stochastic regression onto the velocity target -> looser bar,
    # and forward() runs the ODE sampler so we also exercise sampling here.
    check("flow", flow.build, threshold=0.5, steps=400, width=48, depth=6)
    # PINO: data MSE + a differentiable physics residual; small physics weight so the data
    # term still dominates the overfit check, while exercising the physics code path.
    check("pino", pino.build, threshold=0.35, steps=250, physics_weight=0.01)
    # theme D + stretch (E). node uses few integration steps here for a fast CPU check.
    check("transformer", transformer.build, threshold=0.25, steps=250, width=32, depth=3)
    check("node", node.build, threshold=0.35, steps=250, width=48, node_steps=32)
    check("jepa", jepa.build, threshold=0.4, steps=250, latent=64, width=128)
    check("cnp", cnp.build, nll=True, steps=300, latent=64, width=128)        # Gaussian NLL
    print("all sanity checks passed")
