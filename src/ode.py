import torch
import numpy as np
from common.physics import *
from common.settings_crt import *


class ODE:
    """
    System of ordinary differential equations to solve the cosmological radiative
    transfer equations derived in

    [1] Fukugita, M. & Kawasaki, M. 1994, MNRAS 269, 563 and also discussed in
    [2] Krause F., Thomas R. M., Zaroubi S., Abdalla F. B., 2018, NewAst, 64, 9
    """

    def __init__(self, conf):

        # obtain train_set_size from config
        self.train_set_size = conf.train_set_size

        # generate over_densities array
        self.over_densities = torch.ones((conf.train_set_size))

    def compute_ode_residual(self, flux_vector, state_vector, parameter_vector, u_approximation):
        """
        Takes in the input parameters for neural network and returns the residual computed
        by substituting the output of neural network in our system of four differential equations.
        """
        u_prediction = u_approximation(flux_vector, state_vector)

        # unpack the state_vector
        t = state_vector[:, 4]

        # unpack the prediction vector
        x_H_II_prediction, x_He_II_prediction, x_He_III_prediction, T_prediction = u_prediction[:, 0], \
                                                                                   u_prediction[:, 1], \
                                                                                   u_prediction[:, 2], \
                                                                                   u_prediction[:, 3]

        # compute ionisation fractions from the prediction vectors
        x_H_I_prediction = 1.0 - x_H_II_prediction
        x_He_I_prediction = 1.0 - x_He_II_prediction - x_He_III_prediction

        # unpack the parameter vector and obtain redshift
        redshift = parameter_vector[:, 1]

        # initialise the number densities for the H & He ions and the free electrons
        self.init_number_density_vectors(redshift,
                                         x_H_I_prediction,
                                         x_H_II_prediction,
                                         x_He_I_prediction,
                                         x_He_II_prediction,
                                         x_He_III_prediction)

        # obtain loss from each of the equations
        loss_x_H_II = self.get_x_H_II_loss(x_H_I_prediction,
                                           x_H_II_prediction,
                                           T_prediction,
                                           t)

        loss_x_He_II = self.get_x_He_II_loss(x_He_I_prediction,
                                             x_He_II_prediction,
                                             x_He_III_prediction,
                                             T_prediction,
                                             t)

        loss_x_He_III = self.get_x_He_III_loss(x_He_I_prediction,
                                               x_He_II_prediction,
                                               x_He_III_prediction,
                                               T_prediction,
                                               t)

        loss_T = self.get_temperature_loss(x_H_I_prediction,
                                           x_H_II_prediction,
                                           x_He_I_prediction,
                                           x_He_II_prediction,
                                           x_He_III_prediction,
                                           T_prediction,
                                           t)

        return loss_x_H_II + loss_x_He_II + loss_x_He_III + loss_T

    def get_x_H_II_loss(self, x_H_I, x_H_II, T, t):
        """
        Takes in the output of neural network and returns the residual computed
        by substituting the output in the first differential equation for H_II evolution.
        Ref: equation (A.3) in [2], which is a simplified form of equation (26) in [1].
        """
        # hydrogen number density
        n_H = self.n_hydrogen
        # electron number density
        n_e = self.n_e
        # recombination coefficient H_II
        alpha_H_II = self.recombination_H_II(T)
        # [TODO: add beta 1]
        beta1 = 1.0

        # ionsiation rate for H_I, equation (A.6) in [2]
        ionisation_term1 = beta1 * n_e
        ionisation_term2 = torch.FloatTensor(Physics.getInstance().get_ionisation_rate_integral_hydrogen())
        ionisation_rate_H_I = ionisation_term1 + ionisation_term2

        d_xHII_dt = torch.autograd.grad(x_H_II.sum(), t, create_graph=True, allow_unused=True)[0]
        term1 = torch.multiply(ionisation_rate_H_I, x_H_I)
        term2 = torch.multiply(alpha_H_II, torch.divide(torch.square(n_e), n_H))

        return d_xHII_dt - term1 + term2

    def get_x_He_II_loss(self, x_He_I, x_He_II, x_He_III, T, t):
        """
        Takes in the output of neural network and returns the residual computed
        by substituting the output in the second differential equation for He_II evolution.
        Ref: equation (A.4) in [2], a simplified form of equation (29) in [1]
        """
        # electron number density
        n_e = self.n_e
        # collision ionisations for He_I and He_II
        beta_He_I = self.collision_ionisation_He_I(T)
        beta_He_II = self.collision_ionisation_He_II(T)
        # recombination coefficient for He_II and He_III
        alpha_He_II = self.recombination_He_II(T)
        alpha_He_III = self.recombination_He_III(T)
        # dielectric recombination coefficient for He_II
        xi_He_II = self.dielectric_recombination_He_II(T)
        # ionsiation rate for He_I, equation (A.7) in [2]
        ionisation_rate_He_I = torch.FloatTensor(Physics.getInstance().get_ionisation_rate_integral_helium1())

        d_xHeII_dt = torch.autograd.grad(x_He_II.sum(), t, create_graph=True, allow_unused=True)[0]
        term1 = torch.multiply(ionisation_rate_He_I, x_He_I)
        term2 = torch.multiply(beta_He_I, torch.multiply(n_e, x_He_I))
        term3 = torch.multiply(beta_He_II, torch.multiply(n_e, x_He_II))
        term4 = torch.multiply(alpha_He_II, torch.multiply(n_e, x_He_II))
        term5 = torch.multiply(alpha_He_III, torch.multiply(n_e, x_He_III))
        term6 = torch.multiply(xi_He_II, torch.multiply(n_e, x_He_II))

        return d_xHeII_dt - term1 - term2 + term3 + term4 - term5 + term6

    def get_x_He_III_loss(self, x_He_I, x_He_II, x_He_III, T, t):
        """ Takes in the output of neural network and returns the residual computed
        by substituting the output in the third differential equation for He_III evolution.
        Ref: equation (A.5) in [2], a simplified form of equation (30) in [1]
        """
        # electron number density
        n_e = self.n_e
        # recombination coefficient He_III
        alpha_He_III = self.recombination_He_III(T)
        # collision ionisation
        beta_He_II = self.collision_ionisation_He_II(T)
        # ionsiation rate for He_II, equation (A.8) in [2]
        ionisation_rate_He_II = torch.FloatTensor(Physics.getInstance().get_ionisation_rate_integral_helium2())

        d_xHeIII_dt = torch.autograd.grad(x_He_III.sum(), t, create_graph=True, allow_unused=True)[0]
        term1 = torch.multiply(ionisation_rate_He_II, x_He_II)
        term2 = torch.multiply(beta_He_II, torch.multiply(n_e, x_He_II))
        term3 = torch.multiply(alpha_He_III, torch.multiply(n_e, x_He_III))

        return d_xHeIII_dt - term1 - term2 + term3

    def get_temperature_loss(self, x_H_I, x_H_II, x_He_I, x_He_II, x_He_III, T, t):
        """
        Takes in the output of neural network and returns the residual computed
        by substituting the output in the fourth differential equation for electron temperature evolution.
        Ref: equation (A.9) in [2], a simplified form of equation (36) in [1]
        """

        # TODO: move constants to physics.py
        CONSTANT_c = 2.9979e10          # speed of light in cm/s
        CONSTANT_k_B_erg = 1.3807e-16   # Boltzmann constant in erg/K
        CONSTANT_k_B_eV = 8.6173e-5     # Boltzmann constant in eV/K

        n_e = self.n_e  # electron number density
        n_H_I = self.n_H_I
        n_H_II = self.n_H_II
        n_He_I = self.n_He_I
        n_He_II = self.n_He_II
        n_He_III = self.n_He_III

        d_T_dt = torch.autograd.grad(T.sum(), t, create_graph=True, allow_unused=True)[0]

        heating_rate_H_I = torch.ones((self.train_set_size))
        heating_rate_He_I = torch.ones((self.train_set_size))
        heating_rate_He_II = torch.ones((self.train_set_size))

        term_1 = torch.multiply(n_H_I, heating_rate_H_I) \
                 + torch.multiply(n_He_I, heating_rate_He_I) \
                 + torch.multiply(n_He_II, heating_rate_He_II)

        return 4

    def init_number_density_vectors(self, redshift, x_H_I, x_H_II, x_He_I, x_He_II, x_He_III):
        """
        Takes in the redshift and ionisation fractions for H and He and initialises the
        number density variables for all the H and He ionisation fractions. Also,
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
        temperature in the vector. (α2_HII)
        Ref: equation (57) in section B.2 in [1]
        Units of recombination coefficient: cm^3/s
        """
        return 2.6e-13 * torch.pow((temperature_vector/1.e4), -0.8)

    def recombination_He_II(self, temperature_vector):
        """ Takes in the temperature_vector of shape (train_set_size)
        and returns the recombination coefficient for He_II (free → n ≥ 1)
        for each temperature in the vector. (α_He_II)
        Ref: equation (58) in section B.2 in [1]
        Units of recombination coefficient: cm^3/s
        """
        return 1.5e-10 * torch.pow(temperature_vector, -0.6353)

    def recombination_He_III(self, temperature_vector):
        """ Takes in the temperature_vector of shape (train_set_size)
        and returns the recombination coefficient for He_III (free → n ≥ 1)
        for each temperature in the vector. (α_He_III)
        Ref: equation (62) in section B.2 in [1]
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
        Ref: equation (61) in section B.2 in [1]
        Units of recombination coefficient: cm^3/s
        """
        term1 = torch.pow(temperature_vector, -1.5)
        term2 = torch.exp(-4.7e5/temperature_vector)
        term3 = 1 + 0.3*torch.exp(-9.4e4/temperature_vector)

        return 1.90e-3 * term1 * term2 * term3

    def collision_ionisation_He_I(self, temperature_vector):
        """ Takes in the temperature_vector of shape (train_set_size)
        and returns the collision ionisation for He_I (β_HeI) for each
        temperature in the vector.
        Ref: equation (54) in section B.1 in [1]
        Units of recombination coefficient: cm^3/s
        """
        term1 = torch.pow(temperature_vector, 0.5)
        term2 = torch.pow(1 + torch.pow((temperature_vector/1.e5), 0.5), -1.0)
        term3 = torch.exp(-2.853e5/temperature_vector)

        return 2.38e-11 * term1 * term2 * term3

    def collision_ionisation_He_II(self, temperature_vector):
        """ Takes in the temperature_vector of shape (train_set_size)
        and returns the collision ionisation for He_II (β_HeII) for each
        temperature in the vector.
        Ref: equation (55) in section B.1 in [1]
        Units of recombination coefficient: cm^3/s
        """
        term1 = torch.pow(temperature_vector, 0.5)
        term2 = torch.pow(1 + torch.pow((temperature_vector/1.e5), 0.5), -1.0)
        term3 = torch.exp(-6.315e5/temperature_vector)

        return 5.68e-12 * term1 * term2 * term3
