"""
Training entry point for the paper-3 deterministic emulators.

Examples:
  python train.py --model fno      --run_name fno_053      --epochs 300
  python train.py --model deeponet --run_name deeponet_053 --epochs 300 --latent 256
  python train.py --model fno --subset 2000 --epochs 20      # quick local check

Resume after a wall-time kill:
  python train.py --model fno --run_name fno_053 --resume output/fno_053/last.pt
"""
from __future__ import annotations
import os
from config import Config
from data import load_dataset, make_loaders
from trainer import Trainer
from models import build_model


def main(argv=None):
    cfg = Config.from_args(argv)

    tr, va, te, norm = load_dataset(cfg.data_dir, subset=cfg.subset,
                                    seed=cfg.seed, standardize=cfg.standardize)
    trl, val, tel = make_loaders(tr, va, te, batch_size=cfg.batch_size)

    model = build_model(cfg)
    run_dir = os.path.join(cfg.out_dir, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    cfg.save(os.path.join(run_dir, "config.json"))

    n_par = sum(p.numel() for p in model.parameters())
    print(f"model={cfg.model} params={n_par:,} | train/val/test={len(tr)}/{len(va)}/{len(te)}")
    Trainer(model, cfg, normalizer=norm).train(trl, val)


if __name__ == "__main__":
    main()
