"""
Differentiable (torch) physics terms for the PINO track.

Design note -- why not reuse ML-RT2/src/ode/ode_system.py directly:
  * that module implements the *time-evolution* PINN residual: network(flux, state, time)
    -> next state, residual = local ODE at that (state, time). It needs a time axis and the
    full multifrequency flux vector N(E); our paper-3 emulator predicts the whole radial
    profile at the fixed source age from the 8 parameters, with no time axis and no per-energy
    flux. So that residual cannot be applied here (and it is the formulation that did not
    converge before).
  * What IS reusable and STARDUST-consistent: the physical constants and the torch
    recombination-coefficient fits (Fukugita & Kawasaki 1994, section B.2). We port those and
    build a *spatial* residual suited to a profile operator.

The spatial residual is a grey (frequency-integrated), hydrogen photo-ionisation equilibrium:
in the ionised zone, photo-ionisations balance recombinations,
        Gamma_HI(r) (1 - x_HII)  =  alpha_HII(T) x_HII^2   (per n_H, grey),
with an attenuated flux  Gamma_HI(r) ∝ Gamma0 * exp(-tau(r)) / r^2  and optical depth
        tau(r) = kappa * ∫_0^r (1 - x_HII) dr'.
Gamma0(theta), kappa(theta) are small per-sample amplitudes predicted from the parameters
(the exact source luminosity and gas density are absorbed into them); the *shape* of the law
-- attenuation + equilibrium -- is what regularises the operator, especially at the front.
This is an explicit soft prior, not the exact multifrequency non-equilibrium physics.
"""
from __future__ import annotations
import torch

# --- constants (ported from ML-RT2/common/physics_constants.py; STARDUST-consistent) ---
N_H_0 = 1.9e-7          # H number density at z=0 [cm^-3]
N_HE_0 = 1.5e-8         # He number density at z=0 [cm^-3]
E_ION_H = 13.6057       # HI ionization edge [eV]
E_ION_HE1 = 24.5874
E_ION_HE2 = 54.4228
Y_HE = N_HE_0 / N_H_0   # He/H number ratio (~0.079)


# --- recombination coefficients [cm^3/s] (torch; Fukugita & Kawasaki 1994, B.2) ---
def alpha_HII(T):
    return 2.6e-13 * torch.pow(T / 1.0e4, -0.8)

def alpha_HeII(T):
    return 1.5e-10 * torch.pow(T, -0.6353)

def alpha_HeIII(T):
    t = torch.clamp(T, min=1.0)
    return 3.36e-10 * torch.pow(t, -0.5) * torch.pow(t / 1.0e3, -0.2) \
        * torch.pow(1 + torch.pow(t / 4.0e6, 0.7), -1.0)


# --- boundedness: fractions in [0,1], HeII+HeIII <= 1, T > 0 ---
def boundedness_penalty(xHII, xHeII, xHeIII, T):
    def below0(x): return torch.relu(-x) ** 2
    def above1(x): return torch.relu(x - 1.0) ** 2
    pen = below0(xHII) + above1(xHII) + below0(xHeII) + below0(xHeIII)
    pen = pen + torch.relu(xHeII + xHeIII - 1.0) ** 2      # He conservation
    pen = pen + torch.relu(-T + 1.0) ** 2                  # T > ~1 K
    return pen.mean()


# --- front monotonicity: ionised fractions should not increase outward ---
def monotonicity_penalty(xHII, xHeIII):
    def inc(x):                                            # positive outward gradient is unphysical
        return torch.relu(x[..., 1:] - x[..., :-1]) ** 2
    return (inc(xHII) + inc(xHeIII)).mean()


# --- grey hydrogen photo-ionisation equilibrium residual (scale-free) ---
def hydrogen_equilibrium_residual(xHII, T, gamma0, kappa, r_grid):
    """
    xHII, T: (B, L) physical.  gamma0, kappa: (B,) positive amplitudes.  r_grid: (L,) in (0,1].
    Returns mean squared *normalised* residual (in [0,1]), so it is insensitive to overall scale.
    """
    L = xHII.shape[-1]
    dr = 1.0 / L
    xHI = torch.clamp(1.0 - xHII, min=0.0)
    tau = kappa[:, None] * torch.cumsum(xHI, dim=-1) * dr              # (B, L)
    r2 = r_grid[None, :] ** 2 + 1e-4
    lhs = gamma0[:, None] * torch.exp(-tau) / r2 * xHI                 # photo-ionisation
    rhs = alpha_HII(torch.clamp(T, min=1.0)) * xHII ** 2              # recombination
    resid = (lhs - rhs) / (lhs + rhs + 1e-30)                          # normalised, in [-1,1]
    return (resid ** 2).mean()
