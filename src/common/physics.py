import numpy as np
try:
    from common.utils import *
except ImportError:
    from utils import *
try:
    from ..sed import sed_numba as sed_generator
except ImportError:
    from sed import sed_numba as sed_generator


CONSTANT_T_CMB_0 = 2.731                    # CMB temperature at redshift z=0 in Kelvin
CONSTANT_z_T_kin_EQ_T_CMB = 250             # redshift at which gas and CMB temperature were in equilibrium


CONSTANT_n_H_0 = 1.9e-7                     # H_I + H_II number density at z=0 in cm^-3
CONSTANT_n_He_0 = 1.5e-8                    # He_I + He_II + He_III number density at z=0 in cm^-3

CONSTANT_kpc_to_cm = 3.086e21               # kpc in cm


IONIZATION_ENERGY_HYDROGEN = 13.6057        # unit is eV
IONIZATION_ENERGY_HELIUM1 = 24.5874         # unit is eV
IONIZATION_ENERGY_HELIUM2 = 54.4228         # unit is eV

class Physics:
    "A class to handle all the pre-computed physics CPU calculation"

    __instance = None

    @staticmethod
    def getInstance():
        """ Static access method. """
        if Physics.__instance == None:
            Physics()
        return Physics.__instance

    def __init__(self):
        """ private constructor. """
        if Physics.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            Physics.__instance = self

        self.energy_vector = None
        self.sigma_HI = None
        self.sigma_HeI = None
        self.sigma_HeII = None

    def get_photo_ionisation_cross_section_hydrogen():
        if self.energy_vector is not None:
            return self.sigma_HI
        else:
            raise Exception('Energy vector not set!')

    def get_photo_ionisation_cross_section_helium1():
        if self.energy_vector is not None:
            return self.sigma_HeI
        else:
            raise Exception('Energy vector not set!')

    def get_photoionisation_cross_section_helium2():
        if self.energy_vector is not None:
            return self.sigma_HeII
        else:
            raise Exception('Energy vector not set!')

    def get_tau(self, sim_data, E, index):
        """
        Provided the simulation grid data, a given photon energy E, and the index for the computing arrays,
        compute the optical depth tau
        """

        sigma_HI = self.physics_ionisation_cross_section_hydrogen(E)
        sigma_HeI = self.physics_ionisation_cross_section_helium1(E)
        sigma_HeII = self.physics_ionisation_cross_section_helium2(E)

        # [delta r] = kpc, [number density] = cm^-3, [sigma] = cm^2
        tau_HI = sigma_HI * sim_data.delta_radius * np.sum(sim_data.n_H_I[0:index]) * CONSTANT_kpc_to_cm
        tau_HeI = sigma_HeI * sim_data.delta_radius * np.sum(sim_data.n_He_I[0:index]) * CONSTANT_kpc_to_cm
        tau_HeII = sigma_HeII * sim_data.delta_radius * np.sum(sim_data.n_He_II[0:index]) * CONSTANT_kpc_to_cm

        return tau_HI + tau_HeI + tau_HeII

    def set_energy_vector(self, energy_vector):
        self.energy_vector = energy_vector

        for E in self.energy_vector:
            sigma_HI.append(self.physics_ionisation_cross_section_hydrogen(E))
            sigma_HeI.append(self.physics_ionisation_cross_section_helium1(E))
            sigma_HeII.append(self.physics_ionisation_cross_section_helium2(E))

        # convert lists to numpy arrays
        self.sigma_HI = np.as_array(sigma_HI)
        self.sigma_HeI = np.as_array(sigma_HeI)
        self.sigma_HeII = np.as_array(sigma_HeII)

    def set_flux_vector(self, flux_vector):
        self.flux_vector = flux_vector


    # def physics_compute_ionisation_rates(self, energy_vector, source_flux, beta_H_I, n_e):
    #     """
    #     Takes in the energy_vector, source_flux vector N(E,r,t) computed for
    #     every energy E in vector energy_vector, electron number density n_e,
    #     collision ionisation beta_H_I and returns ionization rate of H_I, He_I, He_II
    #     """
    #
    #     # basic checks
    #     assert len(tau) == len(intensity_vector), 'length of intensity vector should be equal to the length of tau vector'
    #     assert len(E) == len(intensity_vector), 'length of energy vector should be equal to the length of tau vector'
    #
    #     # e_tau = np.exp(-1 * tau)
    #     # source_flux = np.multiply(intensity_vector, e_tau)
    #
    #
    #
    #     # function values for every E
    #     integrand_H_I = np.divide(np.multiply(sigma_HI, source_flux), energy_vector)
    #     integrand_He_I = np.divide(np.multiply(sigma_HeI, source_flux), energy_vector)
    #     integrand_He_II = np.divide(np.multiply(sigma_HeII, source_flux), energy_vector)
    #
    #     # compute integrals
    #     ionisation_H_I = np.multiply(n_e, beta_H_I) + utils_simpson_integration(integrand_H_I, energy_vector)
    #     ionisation_He_I = utils_simpson_integration(integrand_He_I, energy_vector)
    #     ionisation_He_II = utils_simpson_integration(integrand_He_II, energy_vector)
    #
    #     return ionisation_H_I, ionisation_He_I, ionisation_He_II
    #


    def physics_ionisation_cross_section_hydrogen(self, photon_energy):
        """
        Takes in photon energy in eV, returns hydrogen ionisation cross-section
        for the transition n=1 --> free. Return unit is cm^2.
        Based on Fukugita, M. & Kawasaki, M. 1994, MNRAS 269, 563, equation B13
        """

        if photon_energy < IONIZATION_ENERGY_HYDROGEN:
            return 0.0

        sigma_0 = 1.18e-11

        z_1 = np.sqrt(photon_energy/IONIZATION_ENERGY_HYDROGEN - 1.0)

        fraction = np.exp(-4.0 * np.arctan(z_1) / z_1) / (1.0 - np.exp(-2.0 * np.pi / z_1))

        sigma = sigma_0 * np.power(photon_energy, -4) * fraction

        return sigma



    def physics_ionisation_cross_section_helium1(self, photon_energy):
        """
        Takes in photon energy in eV, returns helium ionisation cross-section
        for the first ionisation of the first electron. Return unit is cm^2.
        Based on Fukugita, M. & Kawasaki, M. 1994, MNRAS 269, 563, equation B15
        """

        if photon_energy < IONIZATION_ENERGY_HELIUM1:
            return 0.0

        sigma_0 = 1.13e-14

        fraction_1 = 1.0 / np.power(photon_energy, 2.05)
        fraction_2 = 9.775 / np.power(photon_energy, 3.05)

        sigma = sigma_0 * (fraction_1 - fraction_2)

        return sigma


    def physics_ionisation_cross_section_helium2(self, photon_energy):
        """
        Takes in photon energy in eV, returns helium ionisation cross-section
        for the first ionisation of the first electron. Return unit is cm^2.
        Based on Fukugita, M. & Kawasaki, M. 1994, MNRAS 269, 563, equation B16
        """

        if photon_energy < IONIZATION_ENERGY_HELIUM2:
            return 0.0

        sigma_0 = 7.55e-10

        z_2 = np.sqrt(photon_energy / IONIZATION_ENERGY_HELIUM2 - 1.0)

        fraction = np.exp(-4.0 * np.arctan(z_2) / z_2) / (1.0 - np.exp(-2.0 * np.pi / z_2))

        sigma = sigma_0 * np.power(photon_energy, -4) * fraction

        return sigma


# def physics_ionisation_cross_section_hydrogen2(photon_energy):
#     """
#     Takes in photon energy in EV, returns hydrogen ionisation cross-section
#     for the transition n=2 --> free. Return unit is cm^2.
#     Based on Fukugita, M. & Kawasaki, M. 1994, MNRAS 269, 563, equation B14
#     """
#
#     if photon_energy < IONIZATION_ENERGY_HYDROGEN:
#         return 0.0
#
#     sigma_0 = 1.08e-13
#
#     z = (photon_energy/IONIZATION_ENERGY_HYDROGEN - 0.25)**2
#
#     numerator = (3.0 + 4.0*z*z) * (5.0 + 4.0*z*z) * np.exp(-4.0 * np.arctan(2*z)/z)
#
#     denominator = np.power(1.0 + 4.0*z*z, 3.0) * (1.0 - np.exp(-2.0 * np.pi / z))
#
#     fraction = numerator / denominator
#
#     sigma = sigma_0 * np.power(photon_energy, -3) * fraction
#
#     return sigma
