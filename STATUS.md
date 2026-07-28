# ML-RT / STARDUST — progress summary

_Shareable overview for the team. Last updated: 2026-07-28._

## STARDUST simulation (code revival)
- Fixed a build-blocking bug so the C code compiles/links with modern toolchains again; verified
  end-to-end on the test cases.
- Real bug fixes + cleanups in the file readers, config handling, and physics I/O (including one
  that had silently rejected valid density profiles); clearer variable names; a ~20% faster
  interpolation loop (measured).
- Made the Strömgren-sphere test a runtime config option instead of a compile flag.
- Modernised setup: rewritten dependency installer, conda env, Apptainer image, docs, and fixed
  the Python SED-generator/plotting scripts to run under current Python/SciPy.

## ML-RT2 — paper 3 (advanced emulators)
Goal: methods barely used in astrophysical radiative transfer yet, kept comparable to papers 1–2,
and studied for _how they optimise_. All prototypes share one data module, trainer, and metrics,
so comparisons reflect the method, not the setup.

### Eight emulator prototypes (all built + sanity-verified)
Runnable identically via `train.py --model <name>`:

| family | model | one-line idea |
|---|---|---|
| neural operators | **FNO**, **DeepONet** | learn parameters → profile as an operator; resolution-free, fast inference |
| physics-informed | **PINO** | operator + a soft radiative-transfer equilibrium constraint (data-anchored; avoids the old PINN's failure) |
| generative | **flow matching** | modern generative model giving calibrated ensembles / error bars, cheap sampling |
| transformer | **transformer** | attention couples the ionisation-front physics and the four species |
| stretch | **Neural-ODE**, **JEPA**, **CNP** | radial "marching" decoder; joint-embedding; per-radius uncertainty |

### Getting the data
- Dataset 053 is fetched from Google Drive via `gdown` (in the conda env): run
  `data/053_data_set/get_data.sh` (a one-time Drive file id must be set at the top of the script).

### Supporting infrastructure
- Automatic hyperparameter search (Optuna, cluster-parallel) + 1-D sensitivity sweeps for the
  "how models optimise" study.
- Analysis + plotting (emulator-vs-STARDUST profiles; training-dynamics curves).
- Cluster-ready: conda env + Apptainer image spanning 3090 → H200 hardware; 4-day-safe
  checkpoint/resume.

### Documents to look at (incl. non-coders)
- `docs/methods.pdf` — methods brief: what each method is, the key equation, why it's interesting
  for us, references, a pipeline diagram, and a schematic per architecture.
- `diagrams/*.pdf` — standalone schematic for each architecture (same style as the paper 1–2 figures).

## Status
All eight prototypes verified locally; ready for the GPU cluster. Deferred by choice: the
`table_ion.c` HeII physics fix (awaiting paper/author review) and extending the PINO residual
(helium / heating terms).
