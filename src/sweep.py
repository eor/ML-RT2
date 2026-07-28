"""
Hyperparameter search (Optuna + Hyperband/ASHA pruning) and 1-D sensitivity sweeps.

Parallelise across the heterogeneous cluster by pointing many workers at the same
--storage (each node pulls trials from the shared study; ASHA early-stops weak ones):

  # run on several nodes at once (fast no-wait 3090s are ideal for exploration):
  python sweep.py --model fno --study fno_053 --storage sqlite:///sweeps/fno.db \
      --trials 40 --epochs 60 --subset 8000

Then refine the best config at full budget on a larger card and multi-seed with train.py.

Sensitivity sweep -- the 'how do models optimise' figures (train once per value, all else fixed):

  python sweep.py --model fno --sensitivity modes --values 16,24,32,48,64 --epochs 80

Requires: pip install optuna
"""
from __future__ import annotations
import argparse, os, json
from dataclasses import replace
from config import Config
from data import load_dataset, make_loaders
from models import build_model
from trainer import Trainer


def suggest(trial, model):
    """Small, architecture-appropriate search spaces (see the methods brief)."""
    hp = dict(
        lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
    )
    if model == "fno":
        hp.update(width=trial.suggest_categorical("width", [32, 48, 64, 96]),
                  modes=trial.suggest_categorical("modes", [16, 24, 32, 48, 64]),
                  depth=trial.suggest_int("depth", 3, 6))
    elif model == "deeponet":
        hp.update(latent=trial.suggest_categorical("latent", [64, 128, 256, 512]),
                  width=trial.suggest_categorical("width", [128, 256, 512]),
                  depth=trial.suggest_int("depth", 3, 6),
                  n_freqs=trial.suggest_categorical("n_freqs", [32, 64, 128]),
                  ff_sigma=trial.suggest_float("ff_sigma", 1.0, 30.0))       # front bandwidth
    elif model == "flow":
        hp.update(width=trial.suggest_categorical("width", [48, 96, 128]),
                  depth=trial.suggest_int("depth", 6, 12),
                  eval_steps=trial.suggest_categorical("eval_steps", [10, 20, 50]))
    else:
        raise ValueError(f"no search space for model '{model}'")
    return hp


def _data(base):
    tr, va, te, norm = load_dataset(base.data_dir, subset=base.subset, seed=base.seed)
    trl, val, _ = make_loaders(tr, va, te, batch_size=base.batch_size)
    return trl, val, norm


def run_search(args):
    import optuna
    base = Config(model=args.model, data_dir=args.data_dir, subset=args.subset,
                  epochs=args.epochs, device=args.device, amp=args.amp,
                  early_stop_patience=args.epochs)      # let the pruner, not patience, stop trials
    trl, val, norm = _data(base)

    def objective(trial):
        cfg = replace(base, **suggest(trial, args.model),
                      out_dir=os.path.join(args.out_dir, args.study),
                      run_name=f"trial_{trial.number}", ckpt_every=10 ** 9)
        trainer = Trainer(build_model(cfg), cfg, normalizer=norm)

        def cb(epoch, v):
            trial.report(v["mse"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        trainer.train(trl, val, on_epoch_end=cb)
        return trainer.best_val

    # make sure a sqlite:///path/to.db parent directory exists
    if args.storage and args.storage.startswith("sqlite:///"):
        db_path = args.storage[len("sqlite:///"):]
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

    pruner = optuna.pruners.HyperbandPruner(min_resource=5, max_resource=args.epochs,
                                            reduction_factor=3)
    study = optuna.create_study(direction="minimize", study_name=args.study,
                                storage=args.storage, load_if_exists=True, pruner=pruner)
    study.optimize(objective, n_trials=args.trials)

    print("best value:", study.best_value)
    print("best params:", study.best_params)
    out = os.path.join(args.out_dir, args.study)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "best_params.json"), "w") as f:
        json.dump({"value": study.best_value, "params": study.best_params}, f, indent=2)


def run_sensitivity(args):
    def parse(v):
        return float(v) if ("." in v or "e" in v.lower()) else int(v)
    vals = [parse(v) for v in args.values.split(",")]
    base = Config(model=args.model, data_dir=args.data_dir, subset=args.subset,
                  epochs=args.epochs, device=args.device, amp=args.amp)
    trl, val, norm = _data(base)

    results = {}
    for v in vals:
        cfg = replace(base, **{args.sensitivity: v},
                      out_dir=os.path.join(args.out_dir, f"sens_{args.sensitivity}"),
                      run_name=f"{args.sensitivity}_{v}")
        t = Trainer(build_model(cfg), cfg, normalizer=norm)
        t.train(trl, val)
        results[str(v)] = t.best_val
        print(f"{args.sensitivity}={v}: best val {t.best_val:.4e}", flush=True)
    print("sensitivity result:", results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data_dir", default="../data/053_data_set")
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", type=lambda s: s.lower() in ("1", "true", "yes"), default=True)
    ap.add_argument("--out_dir", default="output")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--study", default="study")
    ap.add_argument("--storage", default=None)                # e.g. sqlite:///sweeps/fno.db
    ap.add_argument("--sensitivity", default=None)            # a single Config field name
    ap.add_argument("--values", default=None)                 # comma-separated
    args = ap.parse_args()
    run_sensitivity(args) if args.sensitivity else run_search(args)


if __name__ == "__main__":
    main()
