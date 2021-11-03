import torch
import numpy as np


class CRT:
    """
    Cosmological radiative transfer equations based on Fukugita and Kawasaki 1994:
    "Reheating during Hierarchical Clustering in the Universe Dominated by
    the Cold Dark Matter"
    """

    def __init__(self):
        pass

    def pde(self, x, t, u_approximation):
        u = u_approximation(x, r)

        return pde

    def get_H_II_loss(self, x_H_I, x_H_II, t):
        n_H = 0.0  # hydrogen density
        n_e = 0.0  # electron number density
        alpha_H_II = 0.0  # recombination H_II
        ionisation_rate_H_I = 0.0  # calculate that big integral (A.6)

        dxHII_dt = torch.autograd.grad(x_H_II.sum(), t, create_graph=True)[0]
        term1 = torch.multiply(ionisation_rate_H_I, x_H_I)
        term2 = torch.multiply(alpha_H_II, torch.divide(torch.square(n_e), n_H))

        return dxHII_dt - term1 + term2

    def get_He_II_loss(self, x_He_I, x_He_II, x_He_III, t):
        n_e = 0.0  # electron number density
        beta_He_I = 0.0  # collision ionisation
        beta_He_II = 0.0  # collision ionisation
        alpha_He_II = 0.0  # recombination He_II
        alpha_He_III = 0.0  # recombination He_III
        Xi_He_II = 0.0  # dielectronic recombination He_II

        ionisation_rate_He_I = 0.0  # calculate that big integral (A.7)

        dxHeII_dt = torch.autograd.grad(x_He_II.sum(), t, create_graph=True)[0]
        term1 = torch.multiply(ionisation_rate_He_I, x_He_I)
        term2 = torch.multiply(beta_He_I, torch.multiply(n_e, x_He_I))
        term3 = torch.multiply(beta_He_II, torch.multiply(n_e, x_He_II))
        term4 = torch.multiply(alpha_He_II, torch.multiply(n_e, x_He_II))
        term5 = torch.multiply(alpha_He_III, torch.multiply(n_e, x_He_III))
        term6 = torch.multiply(Xi_He_II, torch.multiply(n_e, x_He_II))

        return dxHeII_dt - term1 - term2 + term3 + term4 - term5 + term6

    def get_He_III_loss(self, x_He_I, x_He_II, x_He_III, t):
        n_e = 0.0  # electron number density
        alpha_He_III = 0.0  # recombination He_III
        beta_He_II = 0.0  # collision ionisation

        ionisation_rate_He_II = 0.0  # calculate that big integral (A.8)

        dxHeIII_dt = torch.autograd.grad(x_He_III.sum(), t, create_graph=True)[0]
        term1 = torch.multiply(ionisation_rate_He_II, x_He_II)
        term2 = torch.multiply(beta_He_II, torch.multiply(n_e, x_He_II))
        term3 = torch.multiply(alpha_He_III, torch.multiply(n_e, x_He_III))

        return dxHeIII_dt - term1 - term2 + term3

    def get_temprature(self):
