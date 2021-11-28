import torch
import numpy as np
from numpy import pi
from common.physics import *
from common.physics_constants import *
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

    def compute_ode_residual(self, flux_vector, state_vector, time_vector, parameter_vector, u_approximation):
        """
        Takes in the input parameters for neural network and returns the residual computed
        by substituting the output of neural network in our system of four differential equations.
        """

        u_0, u_1, u_2, u_3 = u_approximation(flux_vector, state_vector, time_vector)

        x_H_II_prediction = u_0
        x_He_II_prediction = u_1
        x_He_III_prediction = u_2
        T_prediction = u_3

        # compute ionisation fractions from the prediction vectors
        x_H_I_prediction = 1.0 - x_H_II_prediction
        x_He_I_prediction = 1.0 - x_He_II_prediction - x_He_III_prediction

        # unpack the parameter vector and obtain redshift
        self.redshift = parameter_vector[:, 1]
        self.redshift_pow_3 = torch.pow(1.0 + self.redshift, 3)

        # initialise the number densities for the H & He ions and the free electrons
        self.init_number_density_vectors(x_H_I_prediction,
                                         x_H_II_prediction,
                                         x_He_I_prediction,
                                         x_He_II_prediction,
                                         x_He_III_prediction)

        # obtain loss from each of the equations
        loss_x_H_II = self.get_x_H_II_loss(x_H_I_prediction,
                                           x_H_II_prediction,
                                           T_prediction,
                                           time_vector)

        loss_x_He_II = self.get_x_He_II_loss(x_He_I_prediction,
                                             x_He_II_prediction,
                                             x_He_III_prediction,
                                             T_prediction,
                                             time_vector)

        loss_x_He_III = self.get_x_He_III_loss(x_He_I_prediction,
                                               x_He_II_prediction,
                                               x_He_III_prediction,
                                               T_prediction,
                                               time_vector)

        loss_T = self.get_temperature_loss(x_H_I_prediction,
                                           x_H_II_prediction,
                                           x_He_I_prediction,
                                           x_He_II_prediction,
                                           x_He_III_prediction,
                                           T_prediction,
                                           time_vector)

        return loss_x_H_II + loss_x_He_II + loss_x_He_III + loss_T

    def init_number_density_vectors(self, x_H_I, x_H_II, x_He_I, x_He_II, x_He_III):
        """
        This method initialises the number density variables
        Inputs: all ionisation fractions of H and He
        Units: cm^-3
        """
        density_factor = torch.multiply(self.over_densities, self.redshift_pow_3)
        self.n_hydrogen = density_factor * CONSTANT_n_H_0
        self.n_helium = density_factor * CONSTANT_n_He_0

        # update number densities
        self.n_H_I = torch.multiply(self.n_hydrogen, x_H_I)
        self.n_H_II = torch.multiply(self.n_hydrogen, x_H_II)

        self.n_He_I = torch.multiply(self.n_helium, x_He_I)
        self.n_He_II = torch.multiply(self.n_helium, x_He_II)
        self.n_He_III = torch.multiply(self.n_helium, x_He_III)

        # electron number density = sum of number densities of ionised H, He and doubly ionised He
        self.n_e = self.n_H_II + self.n_He_II + 2 * self.n_He_III

    def get_x_H_II_loss(self, x_H_I, x_H_II, T, t):
        """
        Takes in the output of neural network and returns the residual computed
        by substituting the output in the first differential equation for H_II evolution.
        Ref: equation (A.3) in [2], which is a simplified form of equation (26) in [1].
        """
        # hydrogen number density (1/cm^3)
        n_H = self.n_hydrogen
        # electron number density (1/cm^3)
        n_e = self.n_e
        # recombination coefficient H_II (cm^3/s)
        alpha_H_II = self.recombination_H_II(T)
        # collision_ionisation for hydrogen (cm^3/s)
        beta1 = self.collision_ionisation_H_I(T)

        # ionisation rate for H_I, equation (A.6) in [2] (1/s)
        # out unit: (1/s)
        ionisation_term1 = torch.multiply(beta1, n_e)
        ionisation_term2 = torch.FloatTensor(Physics.getInstance().get_ionisation_rate_integral_hydrogen())
        # out unit: (1/s)
        # import pdb; pdb.set_trace()

        ionisation_rate_H_I = ionisation_term1 + ionisation_term2

        # out unit: (1/s)
        d_xHII_dt = torch.autograd.grad(x_H_II.sum(), t, create_graph=True)[0]
        # out unit: (1/s)
        d_xHII_dt = torch.squeeze(d_xHII_dt)
        term1 = torch.multiply(ionisation_rate_H_I, x_H_I)
        # out unit: (1/s)
        term2 = torch.multiply(alpha_H_II, torch.divide(torch.square(n_e), n_H))

        # temporarily here....

        # print("ionisation_term1", ionisation_term1)
        # print("ionisation_term2", ionisation_term2)
        # print("ionisation_rate_H_I", ionisation_rate_H_I)
        # print("x_H_I", x_H_I)
        # print("x_H_II", x_H_II)
        # print("alpha_H_II", alpha_H_II)
        # print("n_e", n_e)
        # print("n_H", n_H)
        # print("beta1", beta1)
        # print("T", T)
        # print("term1", term1)
        # print("term2", term2)

        return (d_xHII_dt - term1 + term2) / MYR_to_SEC

        # TODO: all get*_loss functions need better doc strings

    def get_x_He_II_loss(self, x_He_I, x_He_II, x_He_III, T, t):
        """
        Takes in the output of neural network and returns the residual computed
        by substituting the output in the second differential equation for He_II evolution.
        Ref: equation (A.4) in [2], a simplified form of equation (29) in [1]
        """
        # electron number density (1/cm^3)
        n_e = self.n_e

        # collisional ionisation coefficient for He_I and He_II (cm^3/s)
        beta_He_I = self.collision_ionisation_He_I(T)
        beta_He_II = self.collision_ionisation_He_II(T)

        # recombination coefficient for He_II and He_III (cm^3/s)
        alpha_He_II = self.recombination_He_II(T)
        alpha_He_III = self.recombination_He_III(T)

        # dielectric recombination coefficient for He_II (cm^3/s)
        xi_He_II = self.dielectric_recombination_He_II(T)

        # ionisation rate for He_I, equation (A.7) in [2] (1/s)
        ionisation_rate_He_I = torch.FloatTensor(Physics.getInstance().get_ionisation_rate_integral_helium1())

        # out unit: 1/s
        d_xHeII_dt = torch.autograd.grad(x_He_II.sum(), t, create_graph=True)[0]
        d_xHeII_dt = torch.squeeze(d_xHeII_dt)
        term1 = torch.multiply(ionisation_rate_He_I, x_He_I)
        # out unit: 1/s
        term2 = torch.multiply(beta_He_I, torch.multiply(n_e, x_He_I))
        # out unit: 1/s
        term3 = torch.multiply(beta_He_II, torch.multiply(n_e, x_He_II))
        # out unit: 1/s
        term4 = torch.multiply(alpha_He_II, torch.multiply(n_e, x_He_II))
        # out unit: 1/s
        term5 = torch.multiply(alpha_He_III, torch.multiply(n_e, x_He_III))
        # out unit: 1/s
        term6 = torch.multiply(xi_He_II, torch.multiply(n_e, x_He_II))

        return (d_xHeII_dt - term1 - term2 + term3 + term4 - term5 + term6) / MYR_to_SEC

    def get_x_He_III_loss(self, x_He_I, x_He_II, x_He_III, T, t):
        """
        Takes in the output of neural network and returns the residual computed
        by substituting the output in the third differential equation for He_III evolution.
        Ref: equation (A.5) in [2], a simplified form of equation (30) in [1]
        """
        # electron number density (1/cm^3)
        n_e = self.n_e

        # recombination coefficient He_III (cm^3/s)
        alpha_He_III = self.recombination_He_III(T)

        # collision ionisation (cm^3/s)
        beta_He_II = self.collision_ionisation_He_II(T)

        # ionisation rate for He_II, equation (A.8) in [2] (1/s)
        ionisation_rate_He_II = torch.FloatTensor(Physics.getInstance().get_ionisation_rate_integral_helium2())

        # out unit: 1/s
        d_xHeIII_dt = torch.autograd.grad(x_He_III.sum(), t, create_graph=True)[0]
        d_xHeIII_dt = torch.squeeze(d_xHeIII_dt)
        # out unit: 1/s
        term1 = torch.multiply(ionisation_rate_He_II, x_He_II)
        # out unit: 1/s
        term2 = torch.multiply(beta_He_II, torch.multiply(n_e, x_He_II))
        # out unit: 1/s
        term3 = torch.multiply(alpha_He_III, torch.multiply(n_e, x_He_III))

        return (d_xHeIII_dt - term1 - term2 + term3) / MYR_to_SEC

    def get_temperature_loss(self, x_H_I, x_H_II, x_He_I, x_He_II, x_He_III, T, t):
        """
        Takes in the output of neural network
        Returns: the residual computed by substituting the output in the fourth differential equation for electron
        temperature evolution.
        Ref: equation (A.9) in [2], a simplified form of equation (36) in [1]
        """

        mu = 1.24
        n_H_n_B_ratio = 1.0/(1.0 + 4*(0.15/1.9))
        n_He_n_B_ratio = 1.0/((1.9/0.15) + 4)

        n_H = self.n_hydrogen       # hydrogen number density (1/cm^3)
        n_B = n_H / n_H_n_B_ratio   # baryon number density
        n_e = self.n_e              # electron number density
        n_H_I = self.n_H_I          # more number densities
        n_H_II = self.n_H_II
        n_He_I = self.n_He_I
        n_He_II = self.n_He_II
        n_He_III = self.n_He_III

        d_T_dt = torch.autograd.grad(T.sum(), t, create_graph=True)[0]
        d_T_dt = torch.squeeze(d_T_dt)

        # heating rate integrals
        heating_rate_H_I = torch.FloatTensor(Physics.getInstance().get_heating_rate_integral_hydrogen())
        heating_rate_He_I = torch.FloatTensor(Physics.getInstance().get_heating_rate_integral_helium1())
        heating_rate_He_II = torch.FloatTensor(Physics.getInstance().get_heating_rate_integral_helium2())
        print(heating_rate_H_I)
        term_1 = torch.multiply(n_H_I, heating_rate_H_I)
        term_1 += torch.multiply(n_He_I, heating_rate_He_I)
        term_1 += torch.multiply(n_He_II, heating_rate_He_II)

        # term 2
        zeta_H_I = self.collisional_ionisation_cooling_H_I(T)
        zeta_He_I = self.collisional_ionisation_cooling_He_I(T)
        zeta_He_II = self.collisional_ionisation_cooling_He_II(T)

        term_2 = torch.multiply(zeta_H_I, torch.multiply(n_e, n_H_I))
        term_2 += torch.multiply(zeta_He_I, torch.multiply(n_e, n_He_I))
        term_2 += torch.multiply(zeta_He_II, torch.multiply(n_e, n_He_II))

        # term 3
        eta_H_II = self.recombination_cooling_coefficient_H_II(T)
        eta_He_II = self.recombination_cooling_coefficient_He_II(T)
        eta_He_III = self.recombination_cooling_coefficient_He_III(T)

        term_3 = torch.multiply(eta_H_II, torch.multiply(n_e, n_H_II))
        term_3 += torch.multiply(eta_He_II, torch.multiply(n_e, n_He_II))
        term_3 += torch.multiply(eta_He_III, torch.multiply(n_e, n_He_III))

        # term 4
        omega_He_II = self.recombination_cooling_coefficient_He_II_dielectronic(T)
        term_4 = torch.multiply(omega_He_II, torch.multiply(n_e, n_He_III))

        # term 5
        psi_H_I = self.collisional_excitation_cooling_H_I(T)
        psi_He_I = self.collisional_excitation_cooling_He_I(T)
        psi_He_II = self.collisional_excitation_cooling_He_II(T)

        term_5 = torch.multiply(psi_H_I, torch.multiply(n_e, n_H_I))
        term_5 += torch.multiply(psi_He_I, torch.multiply(n_e, n_He_I))
        term_5 += torch.multiply(psi_He_II, torch.multiply(n_e, n_He_II))

        # term 6
        term_6 = self.compton_cooling_rate(T)

        # term 7
        theta_ff = self.free_free_cooling_coefficient(T)
        number_density_sum = n_H_II + n_He_II + 4 * n_He_III
        term_7 = torch.multiply(theta_ff, torch.multiply(number_density_sum, n_e))

        # term 8
        a_dot_a = self.hubble_parameter()
        term_8 = 7.5 * torch.multiply(a_dot_a, torch.multiply(CONSTANT_BOLTZMANN_EV * T / mu, n_B))

        # the grand total
        total = term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7 + term_8

        # divide total by constants from the left hand side of the temperature equation
        total = total * (2/3) * mu / CONSTANT_BOLTZMANN_EV
        total = torch.divide(total, n_B)

        return (d_T_dt - total) / MYR_to_SEC

    def recombination_H_II(self, temperature_vector):
        """
        Takes in the temperature_vector of shape (train_set_size)
        and returns the recombination coefficient for H_II (free → n ≥ 2) for each
        temperature in the vector. (α2_HII)
        Ref: equation (57) in section B.2 in [1]
        Units of recombination coefficient: cm^3/s
        """
        return 2.6e-13 * torch.pow((temperature_vector/1.e4), -0.8)

    def recombination_He_II(self, temperature_vector):
        """
        Takes in the temperature_vector of shape (train_set_size)
        and returns the recombination coefficient for He_II (free → n ≥ 1)
        for each temperature in the vector. (α_He_II)
        Ref: equation (58) in section B.2 in [1]
        Units of recombination coefficient: cm^3/s
        """
        return 1.5e-10 * torch.pow(temperature_vector, -0.6353)

    def recombination_He_III(self, temperature_vector):
        """
        Takes in the temperature_vector of shape (train_set_size)
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
        """
        Takes in the temperature_vector of shape (train_set_size)
        and returns the dielectric recombination coefficient for He_II (ξ_HeII)
        for each temperature in the vector.
        Ref: equation (61) in section B.2 in [1]
        Units of recombination coefficient: cm^3 s^-1
        """
        term1 = torch.pow(temperature_vector, -1.5)
        term2 = torch.exp(-4.7e5/temperature_vector)
        term3 = 1 + 0.3*torch.exp(-9.4e4/temperature_vector)

        return 1.90e-3 * term1 * term2 * term3

    def collision_ionisation_H_I(self, temperature_vector):
        """
        Takes in the temperature_vector of shape (train_set_size)
        and returns the collision ionisation for H_I (β1_HI) for each
        temperature in the vector.
        Ref: equation (52) in section B.1 in [1]
        Units of recombination coefficient: cm^3/s
        """
        term1 = torch.pow(temperature_vector, 0.5)
        term2 = torch.pow(1 + torch.pow((temperature_vector/1.e5), 0.5), -1.0)
        term3 = torch.exp(-1.578e5/temperature_vector)

        return 5.85e-11 * term1 * term2 * term3

    def collision_ionisation_He_I(self, temperature_vector):
        """
        Takes in the temperature_vector of shape (train_set_size)
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
        """
        Takes in the temperature_vector of shape (train_set_size)
        and returns the collision ionisation for He_II (β_HeII) for each
        temperature in the vector.
        Ref: equation (55) in section B.1 in [1]
        Units of recombination coefficient: cm^3/s
        """
        term1 = torch.pow(temperature_vector, 0.5)
        term2 = torch.pow(1 + torch.pow((temperature_vector/1.e5), 0.5), -1.0)
        term3 = torch.exp(-6.315e5/temperature_vector)

        return 5.68e-12 * term1 * term2 * term3

    def collisional_ionisation_cooling_H_I(self, temperature_vector):
        """
        Takes in temperature of electron vector and returns the hydrogen I collisional ionisation
        cooling coefficient corresponding to each temperature in vector.
        Ref: zeta_H_I in equation (B17) in section B4.1 in [1]
        Units: eV cm^3 s^-1
        """
        term1 = torch.pow(temperature_vector, 0.5)
        term2 = torch.pow(1 + torch.pow(temperature_vector/1e5, 0.5), -1)
        term3 = torch.exp(-1.58e5/temperature_vector)
        return 1.27e-21 * term1 * term2 * term3 * ERG_to_EV

    def collisional_ionisation_cooling_He_I(self, temperature_vector):
        """
        Takes in temperature of electron vector and returns the helium I collisional ionisation
        cooling coefficient corresponding to each temperature in vector.
        Ref: zeta_He_I in equation (B18) in section B4.1 in [1]
        Units: eV cm^3 s^-1
        """
        term1 = torch.pow(temperature_vector, 0.5)
        term2 = torch.pow(1 + torch.pow(temperature_vector/1e5, 0.5), -1)
        term3 = torch.exp(-2.85e5/temperature_vector)
        return 9.38e-22 * term1 * term2 * term3 * ERG_to_EV

    def collisional_ionisation_cooling_He_II(self, temperature_vector):
        """
        Takes in temperature of electron vector and returns the helium II collisional ionisation
        cooling coefficient corresponding to each temperature in vector.
        Ref: zeta_He_II in equation (B20) in section B4.1 in [1]
        Units: eV cm^3 s^-1
        """
        term1 = torch.pow(temperature_vector, 0.5)
        term2 = torch.pow(1 + torch.pow(temperature_vector/1e5, 0.5), -1)
        term3 = torch.exp(-6.31e5/temperature_vector)
        return 4.95e-22 * term1 * term2 * term3 * ERG_to_EV

    def recombination_cooling_coefficient_H_II(self, temperature_vector):
        """
        Input: temperature vector
        Returns: recombination cooling coefficient for H II
        Ref: equation (B21) in section B4.2 in [1]
        Units: eV cm^3 s^-1
        """
        term1 = torch.pow(temperature_vector, 0.5)
        term2 = torch.pow(temperature_vector/1e3, -0.2)
        term3 = torch.pow(1 + torch.pow(temperature_vector/1e6, 0.7), -1)

        return 6.50e-27 * term1 * term2 * term3 * ERG_to_EV

    def recombination_cooling_coefficient_He_II(self, temperature_vector):
        """
        Input: temperature vector
        Returns: recombination cooling coefficient for He II
        Ref: equation (B22) in section B4.2 in [1]
        Units: eV cm^3 s^-1
        """
        term1 = torch.pow(temperature_vector, 0.3647)

        return 1.55e-26 * term1 * ERG_to_EV

    def recombination_cooling_coefficient_He_II_dielectronic(self, temperature_vector):
        """
        Input: temperature vector
        Returns: recombination cooling coefficient for He II (dielectronic recombination)
        Ref: equation (B23) in section B4.2 in [1]
        Units: eV cm^3 s^-1
        """
        term1 = torch.pow(temperature_vector, -1.5)
        term2 = torch.exp(-4.7e5/temperature_vector)
        term3 = 1 + 0.3 * torch.exp(-9.4e4/temperature_vector)

        return 1.24e-13 * term1 * term2 * term3 * ERG_to_EV

    def recombination_cooling_coefficient_He_III(self, temperature_vector):
        """
        Input: temperature vector
        Returns: recombination cooling coefficient for He III
        Ref: equation (B24) in section B4.3 in [1]
        Units: eV cm^3 s^-1
        """
        term1 = torch.pow(temperature_vector, 0.5)
        term2 = torch.pow(temperature_vector/1e3, -0.2)
        term3 = torch.pow(1 + torch.pow(temperature_vector/4e6, 0.7), -1)

        return 3.48e-26 * term1 * term2 * term3 * ERG_to_EV

    def collisional_excitation_cooling_H_I(self, temperature_vector):
        """
        Input: temperature vector
        Returns: collisional-ionisation cooling coefficient for H I
        Ref: equation (B25) in section B4.3 in [1]
        Units: eV cm^3 s^-1
        """
        term1 = torch.pow(1 + torch.pow(temperature_vector/1e5, 0.5), -1)
        term2 = torch.exp(-1.18e5/temperature_vector)
        return 7.5e-19 * term1 * term2 * ERG_to_EV

    def collisional_excitation_cooling_He_I(self, temperature_vector):
        """
        Input: temperature vector
        Returns: collisional-ionisation cooling coefficient for He I
        Ref: equation (B26) in section B4.3 in [1]
        Units: eV cm^3 s^-1
        """
        term1 = torch.pow(temperature_vector, -0.1687)
        term2 = torch.pow(1 + torch.pow(temperature_vector/1e5, 0.5), -1)
        term3 = torch.exp(-1.31e4/temperature_vector)
        return 9.10e-27 * term1 * term2 * term3 * self.n_e * self.n_He_II * ERG_to_EV

    def collisional_excitation_cooling_He_II(self, temperature_vector):
        """
        Input: temperature vector
        Returns: collisional-ionisation cooling coefficient for He II
        Ref: equation (B27) in section B4.3 in [1]
        Units: eV cm^3 s^-1
        """
        term1 = torch.pow(temperature_vector, -0.397)
        term2 = torch.pow(1 + torch.pow(temperature_vector/1e5, 0.5), -1)
        term3 = torch.exp(-4.73e5/temperature_vector)
        return 5.54e-17 * term1 * term2 * term3 * ERG_to_EV

    def free_free_cooling_coefficient(self, temperature_vector):
        """
        Input: temperature vector
        Returns: the free-free cooling coefficient  theta_ff
        Ref: equation (B28) in section B4.4 of [1]
        Units: K^(0.5)? Unit of gaunt factor?
        """
        return 1.42e-27 * 1.1 * torch.pow(temperature_vector, 0.5)

    def compton_cooling_rate(self, temperature_vector):
        """
        Input: temperature vector
        Returns: compton cooling rate.
        Ref: equation (B29) in section B4.4 of [1]
        Units: eV cm^-3 s^-1
        """

        T_gamma = CONSTANT_COSMO_T_CMB_0 * (1 + self.redshift)

        # unit is K
        term1 = temperature_vector - T_gamma

        term2 = (pi**2) / 15

        # unit is cm^(-3)
        term3 = torch.pow(((CONSTANT_BOLTZMANN_EV * T_gamma * 2 * pi)/(CONSTANT_PLANCK_EV * CONSTANT_LIGHT_SPEED)), 3)

        term4 = CONSTANT_BOLTZMANN_ERG * T_gamma / (CONSTANT_MASS_ELECTRON * CONSTANT_LIGHT_SPEED * CONSTANT_LIGHT_SPEED)

        return 4 * CONSTANT_BOLTZMANN_EV * term1 * term2 * term3 * term4 * \
            self.n_e * CONSTANT_THOMSON_ELEC_CROSS * CONSTANT_LIGHT_SPEED

        # TODO re-check units here!!

    def hubble_parameter(self):
        """
        Return the Hubble parameter for the whole batch at the given redshift.
        Units of hubble parameter: 1/s
        """
        term = torch.pow((CONSTANT_COSMOS_OMEGA_M * self.redshift_pow_3) + (1.0 - CONSTANT_COSMOS_OMEGA_M), 0.5)

        return CONSTANT_HUBBLE_Z0 * term * (KM_to_CM/MPC_to_CM)
