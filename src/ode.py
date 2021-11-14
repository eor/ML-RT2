import torch
import numpy as np
from common.physics import *


class ODE:
    """
    System of ordinary differential equations to solve the cosmological radiative
    transfer equations derived in Fukugita, M. & Kawasaki, M. 1994, MNRAS 269, 563
    """

    def __init__(self):
        pass

    def compute_ode_residual(self, sed_vector, state_vector, u_approximation):
        u = u_approximation(sed_vector, state_vector)

        # TODO: fix this
        # TODO: code up equations, add necessary heating and ionisation rates to the physics module
        # TODO: compute residual and pass it to the network
        return u

    def get_x_H_II_loss(self, x_H_I, x_H_II, t):
        n_H = 0.0  # hydrogen density
        n_e = 0.0  # electron number density
        alpha_H_II = 0.0  # recombination H_II
        ionisation_rate_H_I = 0.0  # calculate that big integral (A.6)

        d_xHII_dt = torch.autograd.grad(x_H_II.sum(), t, create_graph=True)[0]
        term1 = torch.multiply(ionisation_rate_H_I, x_H_I)
        term2 = torch.multiply(alpha_H_II, torch.divide(torch.square(n_e), n_H))

        return d_xHII_dt - term1 + term2

    def get_x_He_II_loss(self, x_He_I, x_He_II, x_He_III, t):
        n_e = 0.0  # electron number density
        beta_He_I = 0.0  # collision ionisation
        beta_He_II = 0.0  # collision ionisation
        alpha_He_II = 0.0  # recombination He_II
        alpha_He_III = 0.0  # recombination He_III
        Xi_He_II = 0.0  # dielectronic recombination He_II

        ionisation_rate_He_I = 0.0  # calculate that big integral (A.7)

        d_xHeII_dt = torch.autograd.grad(x_He_II.sum(), t, create_graph=True)[0]
        term1 = torch.multiply(ionisation_rate_He_I, x_He_I)
        term2 = torch.multiply(beta_He_I, torch.multiply(n_e, x_He_I))
        term3 = torch.multiply(beta_He_II, torch.multiply(n_e, x_He_II))
        term4 = torch.multiply(alpha_He_II, torch.multiply(n_e, x_He_II))
        term5 = torch.multiply(alpha_He_III, torch.multiply(n_e, x_He_III))
        term6 = torch.multiply(Xi_He_II, torch.multiply(n_e, x_He_II))

        return d_xHeII_dt - term1 - term2 + term3 + term4 - term5 + term6

    def get_x_He_III_loss(self, x_He_I, x_He_II, x_He_III, t):
        n_e = 0.0  # electron number density
        alpha_He_III = 0.0  # recombination He_III
        beta_He_II = 0.0  # collision ionisation

        ionisation_rate_He_II = 0.0  # calculate that big integral (A.8)

        d_xHeIII_dt = torch.autograd.grad(x_He_III.sum(), t, create_graph=True)[0]
        term1 = torch.multiply(ionisation_rate_He_II, x_He_II)
        term2 = torch.multiply(beta_He_II, torch.multiply(n_e, x_He_II))
        term3 = torch.multiply(alpha_He_III, torch.multiply(n_e, x_He_III))

        return d_xHeIII_dt - term1 - term2 + term3

    def get_temperature_loss(self, x_H_I, x_H_II, x_He_I, x_He_II, x_He_III, t):

        return 4
