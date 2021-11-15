import torch
import numpy as np
from common.physics import *
from common.settings_crt import *


class ODE:
    """
    System of ordinary differential equations to solve the cosmological radiative
    transfer equations derived in Fukugita, M. & Kawasaki, M. 1994, MNRAS 269, 563
    """

    def __init__(self, conf):
        self.CONSTANT_n_H_0 = 1   #TODO: fix this
        self.CONSTANT_n_He_0 = 1  #TODO: fix this

        # obtain train_set_size from config
        self.train_set_size = conf.train_set_size

        # generate over_densities array
        self.over_densities = torch.ones((conf.train_set_size))

    def compute_ode_residual(self, sed_vector, state_vector, parameter_vector, u_approximation):
        u = u_approximation(sed_vector, state_vector)

        # unpack the state_vector
        t = state_vector[:, 4]

        # unpack the prediction vector
        x_H_II_approx, x_He_II_approx, x_He_III_approx, T_approx = u[:, 0], u[:, 1], u[:, 2], u[:, 3]

        # unpack the parameter vector for halo mass and redshift
        self.halo_mass, self.redshift = parameter_vector[:, 0], parameter_vector[:, 1]

        # number density arrays for total hydrogen and helium in units of cm^-3
        # these should be static but static class variables are a headache in Python
        density_factor = self.over_densities * torch.pow(1.0 + self.redshift, 3)
        self.n_hydrogen = density_factor * self.CONSTANT_n_H_0
        self.n_helium = density_factor * self.CONSTANT_n_He_0

        # ionisation fractions
        x_H_I_approx = 1.0 - x_H_II_approx
        x_He_I_approx = 1.0 - x_He_II_approx - x_He_III_approx

        # update number densities
        self.n_H_I = self.n_hydrogen * x_H_I_approx
        self.n_H_II = self.n_hydrogen * x_H_II_approx
        self.n_He_I = self.n_helium * x_He_I_approx
        self.n_He_II = self.n_helium * x_He_II_approx
        self.n_He_III = self.n_helium * x_He_III_approx

        # electron number density = sum of number densities of ionised H, He and doubly ionised He
        self.n_e = self.n_H_II + self.n_He_II + 2 * self.n_He_III

        # compute loss
        loss_x_H_II = self.get_x_H_II_loss(x_H_I_approx, x_H_II_approx, t)
        loss_x_He_II = self.get_x_He_II_loss(x_He_I_approx, x_He_II_approx, x_He_III_approx, t)
        loss_x_He_III = self.get_x_He_III_loss(x_He_I_approx, x_He_II_approx, x_He_III_approx, t)

        loss_T = self.get_temperature_loss(x_H_I_approx, x_H_II_approx,
                                           x_He_I_approx, x_He_II_approx, x_He_III_approx,
                                           T_approx,
                                           t)

        return loss_x_H_II + loss_x_He_II + loss_x_He_III + loss_T

    def get_x_H_II_loss(self, x_H_I, x_H_II, t):
        n_H = self.n_hydrogen  # hydrogen density
        n_e = self.n_e  # electron number density
        alpha_H_II = 0.0  # recombination H_II

        # calculate that big integral (A.6)
        # [TODO: fix this]
        ionisation_rate_H_I = torch.ones((self.train_set_size))

        d_xHII_dt = torch.autograd.grad(x_H_II.sum(), t, create_graph=True, allow_unused=True)[0]
        term1 = torch.multiply(ionisation_rate_H_I, x_H_I)
        term2 = torch.multiply(alpha_H_II, torch.divide(torch.square(n_e), n_H))

        return d_xHII_dt - term1 + term2

    def get_x_He_II_loss(self, x_He_I, x_He_II, x_He_III, t):
        n_e = self.n_e  # electron number density
        beta_He_I = 0.0  # collision ionisation
        beta_He_II = 0.0  # collision ionisation
        alpha_He_II = 0.0  # recombination He_II
        alpha_He_III = 0.0  # recombination He_III
        xi_He_II = 0.0  # dielectronic recombination He_II

        # calculate that big integral (A.7)
        # [TODO: fix this]
        ionisation_rate_He_I = torch.ones((self.train_set_size))

        d_xHeII_dt = torch.autograd.grad(x_He_II.sum(), t, create_graph=True)[0]
        term1 = torch.multiply(ionisation_rate_He_I, x_He_I)
        term2 = torch.multiply(beta_He_I, torch.multiply(n_e, x_He_I))
        term3 = torch.multiply(beta_He_II, torch.multiply(n_e, x_He_II))
        term4 = torch.multiply(alpha_He_II, torch.multiply(n_e, x_He_II))
        term5 = torch.multiply(alpha_He_III, torch.multiply(n_e, x_He_III))
        term6 = torch.multiply(xi_He_II, torch.multiply(n_e, x_He_II))

        return d_xHeII_dt - term1 - term2 + term3 + term4 - term5 + term6

    def get_x_He_III_loss(self, x_He_I, x_He_II, x_He_III, t):
        n_e = self.n_e  # electron number density
        alpha_He_III = 0.0  # recombination He_III
        beta_He_II = 0.0  # collision ionisation

        # calculate that big integral (A.8)
        # [TODO: fix this]
        ionisation_rate_He_II = torch.ones((self.train_set_size))

        d_xHeIII_dt = torch.autograd.grad(x_He_III.sum(), t, create_graph=True)[0]
        term1 = torch.multiply(ionisation_rate_He_II, x_He_II)
        term2 = torch.multiply(beta_He_II, torch.multiply(n_e, x_He_II))
        term3 = torch.multiply(alpha_He_III, torch.multiply(n_e, x_He_III))

        return d_xHeIII_dt - term1 - term2 + term3

    def get_temperature_loss(self, x_H_I, x_H_II, x_He_I, x_He_II, x_He_III, T, t):

        # get required densities
        n_H_I = self.n_H_I
        n_H_II = self.n_H_II
        n_He_I = self.n_He_I
        n_He_II = self.n_He_II
        n_He_III = self.n_He_III
        n_e = self.n_e

        # cooling coefficients

        # TODO solve heating rate integrals

        term_2 = torch.multiply(n_e, n_H_I)


        return 4
