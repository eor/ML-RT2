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

        # obtain train_set_size from config
        self.train_set_size = conf.train_set_size

        # generate over_densities array
        self.over_densities = torch.ones((conf.train_set_size))


    def compute_ode_residual(self, flux_vector, state_vector, parameter_vector, u_approximation):
        u_prediction = u_approximation(flux_vector, state_vector)

        # unpack the state_vector
        t = state_vector[:, 4]
        # unpack the prediction vector
        x_H_II_prediction, x_He_II_prediction, x_He_III_prediction, T_prediction = u_prediction[:,0], u_prediction[:,1], u_prediction[:,2], u_prediction[:,3]
        # compute ionisation fractions from the prediction vectors
        x_H_I_prediction = 1.0 - x_H_II_prediction
        x_He_I_prediction = 1.0 - x_He_II_prediction - x_He_III_prediction

        # unpack the parameter vector and obtain redshift
        redshift = parameter_vector[:, 1]
        # initialse the number density arrays and electron number density arrays
        self.init_number_density_vectors(redshift, x_H_I_prediction, x_H_II_prediction, x_He_I_prediction, x_He_II_prediction, x_He_III_prediction)

        # obtain loss from each of the equations
        loss_x_H_II = self.get_x_H_II_loss(x_H_I_prediction, x_H_II_prediction, T_prediction, t)
        loss_x_He_II = self.get_x_He_II_loss(x_He_I_prediction, x_He_II_prediction, x_He_III_prediction, T_prediction, t)
        loss_x_He_III = self.get_x_He_III_loss(x_He_I_prediction, x_He_II_prediction, x_He_III_prediction, T_prediction, t)
        loss_T = self.get_temperature_loss(x_H_I_prediction, x_H_II_prediction, x_He_I_prediction, x_He_II_prediction, x_He_III_prediction, T_prediction, t)

        return loss_x_H_II + loss_x_He_II + loss_x_He_III + loss_T

    def get_x_H_II_loss(self, x_H_I, x_H_II, T, t):
        n_H = self.n_hydrogen  # hydrogen density
        n_e = self.n_e  # electron number density
        alpha_H_II = self.recombination_H_II(T)  # recombination H_II

        # calculate that big integral (A.6)
        # [TODO: fix this]
        ionisation_rate_H_I = torch.ones((self.train_set_size))

        d_xHII_dt = torch.autograd.grad(x_H_II.sum(), t, create_graph=True, allow_unused=True)[0]
        term1 = torch.multiply(ionisation_rate_H_I, x_H_I)
        term2 = torch.multiply(alpha_H_II, torch.divide(torch.square(n_e), n_H))

        return d_xHII_dt - term1 + term2

    def get_x_He_II_loss(self, x_He_I, x_He_II, x_He_III, T, t):
        n_e = self.n_e  # electron number density
        beta_He_I = 0.0  # collision ionisation
        beta_He_II = 0.0  # collision ionisation
        alpha_He_II = self.recombination_He_II(T)  # recombination He_II
        alpha_He_III = self.recombination_He_III(T)  # recombination He_III
        Xi_He_II = self.dielectric_recombination_He_II(T)  # dielectronic recombination He_II

        # calculate that big integral (A.7)
        # [TODO: fix this]
        ionisation_rate_He_I = torch.ones((self.train_set_size))

        d_xHeII_dt = torch.autograd.grad(x_He_II.sum(), t, create_graph=True, allow_unused=True)[0]
        term1 = torch.multiply(ionisation_rate_He_I, x_He_I)
        term2 = torch.multiply(beta_He_I, torch.multiply(n_e, x_He_I))
        term3 = torch.multiply(beta_He_II, torch.multiply(n_e, x_He_II))
        term4 = torch.multiply(alpha_He_II, torch.multiply(n_e, x_He_II))
        term5 = torch.multiply(alpha_He_III, torch.multiply(n_e, x_He_III))
        term6 = torch.multiply(Xi_He_II, torch.multiply(n_e, x_He_II))

        return d_xHeII_dt - term1 - term2 + term3 + term4 - term5 + term6

    def get_x_He_III_loss(self, x_He_I, x_He_II, x_He_III, T, t):
        n_e = self.n_e  # electron number density
        alpha_He_III = self.recombination_He_III(T)  # recombination He_III
        beta_He_II = 0.0  # collision ionisation

        # calculate that big integral (A.8)
        # [TODO: fix this]
        ionisation_rate_He_II = torch.ones((self.train_set_size))

        d_xHeIII_dt = torch.autograd.grad(x_He_III.sum(), t, create_graph=True, allow_unused=True)[0]
        term1 = torch.multiply(ionisation_rate_He_II, x_He_II)
        term2 = torch.multiply(beta_He_II, torch.multiply(n_e, x_He_II))
        term3 = torch.multiply(alpha_He_III, torch.multiply(n_e, x_He_III))

        return d_xHeIII_dt - term1 - term2 + term3

    def get_temperature_loss(self, x_H_I, x_H_II, x_He_I, x_He_II, x_He_III, T, t):
        # [TODO: complete this]
        return x_H_I/x_H_I

    def init_number_density_vectors(self, redshift, x_H_I, x_H_II, x_He_I, x_He_II, x_He_III):
        """ Takes in the redshift and ionisation fractions for H and He and initialises the
        number density arrays for all the H and He ionisation fractions. Also,
        initialises the electron number density arrays.
        Units of values in computed arrays: cm^-3
        """

        density_factor = self.over_densities * torch.pow(1.0 + redshift, 3)
        self.n_hydrogen = density_factor * CONSTANT_n_H_0
        self.n_helium = density_factor * CONSTANT_n_He_0

        # update number densities
        self.n_H_I = self.n_hydrogen * x_H_I
        self.n_H_II = self.n_hydrogen * x_H_II
        self.n_He_I = self.n_helium * x_He_I
        self.n_He_II = self.n_helium * x_He_II
        self.n_He_III = self.n_helium * x_He_III

        # electron number density = sum of number densities of ionised H, He and doubly ionised He
        self.n_e = self.n_H_II + self.n_He_II + 2 * self.n_He_III


    def recombination_H_II(self, temperature_vector):
        """ Takes in the temperature_vector of shape (train_set_size)
        and returns the recombination coefficient for H_II (free → n ≥ 2) for each
        temperature in the vector.
        Ref: equation (57) in section B.2 in Fukugita1994
        Units of recombination coefficient: cm^3/s
        """
        return 2.6e-13 * torch.pow((temperature_vector/1.e4), -0.8)

    def recombination_He_II(self, temperature_vector):
        """ Takes in the temperature_vector of shape (train_set_size)
        and returns the recombination coefficient for He_II (free → n ≥ 1)
        for each temperature in the vector.
        Ref: equation (58) in section B.2 in Fukugita1994
        Units of recombination coefficient: cm^3/s
        """
        return 1.5e-10 * torch.pow(temperature_vector, -0.6353)

    def recombination_He_III(self, temperature_vector):
        """ Takes in the temperature_vector of shape (train_set_size)
        and returns the recombination coefficient for He_III (free → n ≥ 1)
        for each temperature in the vector.
        Ref: equation (62) in section B.2 in Fukugita1994
        Units of recombination coefficient: cm^3/s
        """
        term1 = torch.pow(temperature_vector, -0.5)
        term2 = torch.pow((temperature_vector/1.e3), -0.2)
        term3 = torch.pow(1 + torch.pow((temperature_vector/4.e6), 0.7), -1.0)
        return 3.36e-10 * term1 * term2 * term3

    def dielectric_recombination_He_II(self, temperature_vector):
        """ Takes in the temperature_vector of shape (train_set_size)
        and returns the dielectric recombination coefficient for He_II (ξ_HeII)
        for each temperature in the vector.
        Ref: equation (61) in section B.2 in Fukugita1994
        Units of recombination coefficient: cm^3/s
        """
        term1 = torch.pow(temperature_vector, -1.5)
        term2 = torch.exp(-4.7e5/temperature_vector)
        term3 = 1 + 0.3*torch.exp(-9.4e4/temperature_vector)
        return 1.90e-3 * term1 * term2 * term3
