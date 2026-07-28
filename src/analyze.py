"""
Load a trained run and produce rudimentary result plots + test metrics.

  python analyze.py --run_dir output/fno_053
  python analyze.py --run_dir output/fno_053 --data_dir /data/053_data_set --n_samples 6

Writes <run_dir>/plots/{profiles,history}.png. To overlay several architectures on
one training-dynamics figure, call plots.plot_history([h1, h2, ...], out, labels=[...]).
"""
from __future__ import annotations
import os, json, argparse
import numpy as np
import torch

from config import Config
from data import load_dataset, make_loaders
from models import build_model
from metrics import evaluate
import plots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--n_samples", type=int, default=4)
    args = ap.parse_args()

    with open(os.path.join(args.run_dir, "config.json")) as f:
        cfg = Config(**json.load(f))
    if args.data_dir:
        cfg.data_dir = args.data_dir
    cfg.device = "cpu"

    tr, va, te, norm = load_dataset(cfg.data_dir, subset=cfg.subset,
                                    seed=cfg.seed, standardize=cfg.standardize)
    _, _, tel = make_loaders(tr, va, te, batch_size=cfg.batch_size)

    model = build_model(cfg)
    ck = torch.load(os.path.join(args.run_dir, args.ckpt), map_location="cpu")
    model.load_state_dict(ck["model"])
    model.eval()

    metrics = evaluate(model, tel, torch.device("cpu"), norm)
    print("test:", {k: round(v, 5) for k, v in metrics.items() if not isinstance(v, dict)})
    print("rel_l2:", {k: round(v, 4) for k, v in metrics["rel_l2"].items()})

    preds, targs, got = [], [], 0
    with torch.no_grad():
        for p, t in tel:
            preds.append(norm.decode_profiles(model(p)))
            targs.append(norm.decode_profiles(t))
            got += p.shape[0]
            if got >= max(args.n_samples, 8):
                break
    preds = np.concatenate(preds, 0)
    targs = np.concatenate(targs, 0)

    pdir = os.path.join(args.run_dir, "plots")
    os.makedirs(pdir, exist_ok=True)
    idx = list(range(min(args.n_samples, preds.shape[0])))
    plots.plot_sample_profiles(preds, targs, idx, os.path.join(pdir, "profiles.png"))
    plots.plot_history(os.path.join(args.run_dir, "history.json"), os.path.join(pdir, "history.png"))
    print("wrote", os.path.join(pdir, "profiles.png"), "and history.png")


if __name__ == "__main__":
    main()
