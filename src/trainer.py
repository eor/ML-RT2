"""
Generic trainer for the *deterministic* emulators (FNO, DeepONet, transformer-op, ...).

Design goals for paper 3:
  * runs unchanged on a 3090 or an H200  -> configurable batch size + grad accumulation,
    bf16 autocast (supported Ampere->Hopper), device auto-detect.
  * survives the 4-day wall-time cap      -> resumable checkpoints (model+optim+sched+epoch).
  * supports the 'how do models optimise' study -> logs per-epoch train/val loss, per-channel
    val metrics, gradient norm and epoch wall-time into a history JSON for cross-model plots.

Generative models (diffusion / flow matching) get their own loop but reuse this history format.
"""
from __future__ import annotations
import os, time, json, math
import torch
from metrics import mse, evaluate


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def amp_dtype(name: str):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


class Trainer:
    def __init__(self, model, cfg, normalizer=None):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.model = model.to(self.device)
        self.normalizer = normalizer
        # physics-aware models (PINO) need the decode stats to evaluate the residual
        if hasattr(self.model, "set_normalizer"):
            self.model.set_normalizer(normalizer)
            self.model.to(self.device)
        self.opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=cfg.epochs)
        self.dtype = amp_dtype(cfg.amp_dtype) if cfg.amp else torch.float32
        self.use_amp = cfg.amp and self.device.type == "cuda"
        # fp16 needs a grad scaler; bf16 does not
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.use_amp and self.dtype == torch.float16))
        self.history = {"train_loss": [], "val": [], "grad_norm": [], "epoch_time": [], "lr": []}
        self.start_epoch = 0
        self.best_val = math.inf
        self.run_dir = os.path.join(cfg.out_dir, cfg.run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        if cfg.resume:
            self.load_checkpoint(cfg.resume)

    # ------------------------------------------------------------------ #
    def _step_loss(self, params, target):
        # generative models (flow matching, diffusion) expose their own objective;
        # deterministic regressors fall back to MSE on the prediction.
        if hasattr(self.model, "training_loss"):
            return self.model.training_loss(params, target)
        return mse(self.model(params), target)

    def train(self, train_loader, val_loader, on_epoch_end=None):
        """on_epoch_end(epoch, val_metrics) is called after each epoch's evaluation
        (used by the Optuna sweep to report intermediate values and prune)."""
        cfg = self.cfg
        patience = 0
        for epoch in range(self.start_epoch, cfg.epochs):
            t0 = time.time()
            self.model.train()
            running, gnorm_acc, nb = 0.0, 0.0, 0
            self.opt.zero_grad(set_to_none=True)
            for it, (params, target) in enumerate(train_loader):
                params, target = params.to(self.device), target.to(self.device)
                with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp):
                    loss = self._step_loss(params, target) / cfg.grad_accum
                self.scaler.scale(loss).backward()
                if (it + 1) % cfg.grad_accum == 0:
                    self.scaler.unscale_(self.opt)
                    gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.clip_grad)
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    self.opt.zero_grad(set_to_none=True)
                    gnorm_acc += float(gnorm); nb += 1
                running += float(loss) * cfg.grad_accum
            self.sched.step()

            val = evaluate(self.model, val_loader, self.device, self.normalizer)
            dt = time.time() - t0
            self.history["train_loss"].append(running / len(train_loader))
            self.history["val"].append(val)
            self.history["grad_norm"].append(gnorm_acc / max(nb, 1))
            self.history["epoch_time"].append(dt)
            self.history["lr"].append(self.opt.param_groups[0]["lr"])
            self._save_history()

            improved = val["mse"] < self.best_val - 1e-6
            if improved:
                self.best_val = val["mse"]; patience = 0
                self.save_checkpoint("best.pt", epoch)
            else:
                patience += 1
            if (epoch + 1) % cfg.ckpt_every == 0:
                self.save_checkpoint("last.pt", epoch)

            extra = f" front={val.get('front_err_cells'):.1f}" if 'front_err_cells' in val else ""
            print(f"[{cfg.run_name}] epoch {epoch:4d} | train {running/len(train_loader):.4e} "
                  f"| val {val['mse']:.4e}{extra} | gnorm {self.history['grad_norm'][-1]:.2f} "
                  f"| {dt:.1f}s", flush=True)

            if on_epoch_end is not None:
                on_epoch_end(epoch, val)        # may raise (e.g. optuna.TrialPruned)

            if patience >= cfg.early_stop_patience:
                print(f"early stopping at epoch {epoch} (no val improvement for {patience})")
                break
        self.save_checkpoint("last.pt", epoch)
        return self.history

    # ------------------------------------------------------------------ #
    def save_checkpoint(self, name, epoch):
        torch.save({
            "epoch": epoch + 1, "best_val": self.best_val,
            "model": self.model.state_dict(), "opt": self.opt.state_dict(),
            "sched": self.sched.state_dict(), "history": self.history,
        }, os.path.join(self.run_dir, name))

    def load_checkpoint(self, path):
        ck = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["opt"])
        self.sched.load_state_dict(ck["sched"])
        self.start_epoch = ck["epoch"]; self.best_val = ck["best_val"]
        self.history = ck.get("history", self.history)
        print(f"resumed from {path} @ epoch {self.start_epoch}")

    def _save_history(self):
        with open(os.path.join(self.run_dir, "history.json"), "w") as f:
            json.dump(self.history, f)

    # ------------------------------------------------------------------ #
    def overfit_batch(self, batch, steps=300):
        """Sanity check: a correct model must drive one batch's loss ~ 0."""
        params, target = batch
        params, target = params.to(self.device), target.to(self.device)
        self.model.train()
        losses = []
        for _ in range(steps):
            self.opt.zero_grad(set_to_none=True)
            loss = self._step_loss(params, target)
            loss.backward(); self.opt.step()
            losses.append(float(loss))
        return losses
