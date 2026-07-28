# Architecture diagrams (paper 3)

Standalone TikZ schematics, one per method, matching the convention of the comparison-paper
diagrams in `ML-RT/diagrams/`. Each `.tex` is a `standalone` document that compiles to its own
cropped PDF for inclusion in the paper (and shareable with non-coders).

Build one:
```bash
latexmk -pdf fno.tex        # -> fno.pdf   (or: pdflatex fno.tex)
```

The high-level *pipeline* diagram (parameters -> emulator -> profiles) lives inside the methods
brief itself (`docs/methods.tex`, Figure 1); these per-architecture figures zoom into each model.

| file | method | status |
|------|--------|--------|
| `fno.tex` | Fourier Neural Operator | done |
| `deeponet.tex` | DeepONet (branch x trunk) | done |
| `pino.tex` | physics-informed operator (+ residual loop) | done |
| `flow.tex` | conditional flow matching | done |
| `transformer.tex` | profile/signal transformer | done |
| `node.tex` | Neural-ODE in radius | done |
| `jepa.tex` | joint-embedding predictive | done |
| `cnp.tex` | conditional neural process | done |
