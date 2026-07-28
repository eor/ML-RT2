# ML-RT paper 3 — advanced emulators for STARDUST

Prototypes for the third ML-RT paper: **methods barely used in astrophysical radiative
transfer yet**, evaluated for comparability against papers 1–2 (MLP / CVAE / CGAN / LSTM +
custom) and studied for **how they optimise**.

This subtree is self-contained and does **not** modify anything in `ML-RT/` or the rest of
`ML-RT2/`.

## The task

STARDUST is a deterministic map:

| input | output |
|---|---|
| 8 physical parameters (log halo mass, redshift, source age, QSO α, QSO efficiency, stellar escape fraction, IMF slope, log IMF min-mass) | 4 radial profiles — `x_HII`, `x_HeII`, `x_HeIII`, `T` — each 1500 points over radius |

Dataset **053** (41,081 samples) is used for direct comparison with papers 1–2. Transforms
match those papers: parameters min-max scaled to `[0,1]`; fraction profiles `log10(x+1e-6)`;
temperature `log10(T)`; we additionally z-score each channel on the train split (stored on the
`Normalizer` for inversion). Split is 80/10/10 at seed 42.

## Architecture plan

The input is only 8 scalars, so the "advanced" payoff is in the **decoder / generative / physics**
side, not attention-over-input. Three frontiers, plus stretch ideas:

| theme | model(s) | why it's a fit / novel for astro-RT | status |
|---|---|---|---|
| **A. Neural operators** | **FNO-1D**, **DeepONet** | outputs are smooth 1D fields with a sharp front; spectral / basis operators are resolution-free and cheap at inference | **prototypes done** |
| **B. Physics-informed operator** | **PINO** | operator + a differentiable grey ionisation-equilibrium residual (ported recombination fits); data-anchored, avoids the old PINN failure; most novel | **prototype done** |
| **C. Generative / probabilistic** | **conditional flow matching** (diffusion later) | successor to CVAE/CGAN; calibrated ensembles + error bars; few-step sampling keeps inference cheap | **prototype done** |
| **D. Attention decoder** | **profile/signal transformer** | long-range coupling (front position ↔ integrated absorption), inter-species attention | planned |
| **E. Stretch** | JEPA-style joint embedding, Neural-ODE-in-r, Neural Processes | very novel for astro; data efficiency / physical structure / lightweight uncertainty | ideas |

All models share the data module, metrics, trainer, and history format, so the cross-architecture
**optimisation study** (loss curves, per-channel val error, gradient norms, epoch wall-time,
sample-efficiency sweeps) is apples-to-apples.

## Layout

```
ML-RT2/                     (this project = paper 3; the earlier PINN work is in archive/)
  environment.yml         conda env (CUDA 12.1; spans 3090 → H200)
  STATUS.md               shareable progress summary
  data/                   dataset 053 / 057 (see data/053_data_set/get_data.sh)
  apptainer/emulators.def     portable image for the heterogeneous cluster
  docs/
    methods.tex/.pdf      shareable methods brief (formulas, refs, pipeline + per-model diagrams)
    refs.bib
  diagrams/
    *.tex/.pdf            standalone per-architecture TikZ schematics (8) (+ README)
  archive/                earlier PINN exploration (legacy_pinn/, study_material/)
  src/
    data.py               053 loader, transforms, Normalizer, synthetic sanity task
    config.py             dataclass config, minimal CLI
    metrics.py            encoded MSE + per-channel + physical rel-L2 + front-position error
    physics.py            torch RT physics for PINO (recombination fits + spatial residual)
    trainer.py            bf16 AMP, grad-accum, checkpoint/resume, diagnostics -> history.json
    train.py              entry point (dispatches by --model)
    sweep.py              Optuna search (Hyperband/ASHA pruning) + 1-D sensitivity sweeps
    analyze.py            load a run -> test metrics + result plots
    plots.py              profile (emulator vs STARDUST) + training-dynamics figures
    sanity_check.py       shape + overfit-a-batch on the synthetic task
    models/
      fno.py              1D Fourier Neural Operator (self-contained)
      deeponet.py         DeepONet with Fourier-feature trunk
      flow.py             conditional flow matching (generative, gives ensembles)
      pino.py             physics-informed neural operator (operator + RT residual)
      transformer.py      profile/signal transformer decoder
      node.py             Neural-ODE decoder in radius
      jepa.py             joint-embedding predictive architecture
      cnp.py              conditional neural process (per-radius uncertainty)
```

## Running

**Local sanity (CPU, no data):**
```bash
cd src && python sanity_check.py          # shape + overfit-a-batch for each model
```

**Get the data** (into `ML-RT2/data/053_data_set`): see `data/053_data_set/get_data.sh`.

**Train (cluster):**
```bash
python train.py --model fno      --run_name fno_053      --epochs 300
python train.py --model deeponet --run_name deeponet_053 --epochs 300 --latent 256
python train.py --model flow     --run_name flow_053     --epochs 400   # generative
python train.py --model pino     --run_name pino_053     --epochs 300 --physics_weight 0.1
# resume after a wall-time kill:
python train.py --model fno --run_name fno_053 --resume output/fno_053/last.pt
```

**Tune (parallel workers share one study):**
```bash
python sweep.py --model fno --study fno_053 --storage sqlite:///sweeps/fno.db \
    --trials 40 --epochs 60 --subset 8000        # run on several nodes at once
python sweep.py --model fno --sensitivity modes --values 16,24,32,48,64 --epochs 80
```

**Analyse a run** (test metrics + plots into `<run_dir>/plots/`):
```bash
python analyze.py --run_dir output/fno_053
```

## Cluster notes
- 4-day wall-time cap → the trainer writes resumable checkpoints (`best.pt`, `last.pt`) and a
  live `history.json`; `--resume` continues seamlessly.
- Heterogeneous GPUs (3090 / L40S / A100 / H200) → bf16 autocast everywhere, and batch size ×
  `--grad_accum` lets the same run fit 24 GB or 141 GB. The Apptainer image pins the CUDA
  userspace so one `.sif` runs on every node.

## Status
FNO, DeepONet, conditional flow-matching, and **PINO** prototypes are wired, overfit-a-batch
verified, and run end-to-end on real 053 data. The Optuna sweep harness (search + sensitivity)
and the analysis/plotting are in place. Next: transformer-operator (D) and the stretch ideas (E),
and extending the PINO residual (helium equilibrium, optional heating term).
