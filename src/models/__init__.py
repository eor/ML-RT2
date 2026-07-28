from . import fno, deeponet, flow, pino, transformer, node, jepa, cnp

# central model registry
BUILDERS = {
    "fno": fno.build,            # A: Fourier neural operator
    "deeponet": deeponet.build,  # A: DeepONet
    "flow": flow.build,          # C: conditional flow matching (generative)
    "pino": pino.build,          # B: physics-informed neural operator
    "transformer": transformer.build,  # D: profile/signal transformer decoder
    "node": node.build,          # E: Neural-ODE decoder in radius
    "jepa": jepa.build,          # E: joint-embedding predictive architecture
    "cnp": cnp.build,            # E: conditional neural process (uncertainty)
}


def build_model(cfg):
    if cfg.model not in BUILDERS:
        raise ValueError(f"unknown model '{cfg.model}'; available: {list(BUILDERS)}")
    return BUILDERS[cfg.model](cfg)
