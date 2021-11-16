import numpy as np
try:
    from common.utils import *
except ImportError:
    from utils import *


CONSTANT_T_CMB_0 = 2.731                    # CMB temperature at redshift z=0 in Kelvin
CONSTANT_z_T_kin_EQ_T_CMB = 250             # redshift at which gas and CMB temperature were in equilibrium


CONSTANT_n_H_0 = 1.9e-7                     # H_I + H_II number density at z=0 in cm^-3
CONSTANT_n_He_0 = 1.5e-8                    # He_I + He_II + He_III number density at z=0 in cm^-3

CONSTANT_kpc_to_cm = 3.086e21               # kpc in cm


IONIZATION_ENERGY_HYDROGEN = 13.6057        # unit is eV
IONIZATION_ENERGY_HELIUM1 = 24.5874         # unit is eV
IONIZATION_ENERGY_HELIUM2 = 54.4228         # unit is eV

class Physics:
    """ A class to handle all the pre-computed physics CPU calculation.

    [1] Fukugita, M. & Kawasaki, M. 1994, MNRAS 269, 563 and also discussed in
    [2] Krause F., Thomas R. M., Zaroubi S., Abdalla F. B., 2018, NewAst, 64, 9
    """

    __instance = None


    @staticmethod
    def getInstance():
        """ Static access method. """
        if Physics.__instance == None:
            Physics()
        return Physics.__instance


    def __init__(self):

        if Physics.__instance != None:
            raise Exception("This class is a singleton. Please access it using Physics.getInstance()")
        else:
            Physics.__instance = self

        # varaibles to store the data set by the user
        self.energy_vector = None
        self.flux_vector = None
        self.reset_pre_computed_variables()


    def get_photo_ionisation_cross_section_hydrogen(self):
        """ Returns the photo ionisation cross section for hydrogen
        if the energy_vector is not None.

        Units:  Takes in energy_vector with enegies in eV
                Return photo_ionisation_cross_section in cm^2
        """
        if self.energy_vector is None:
            raise Exception('Energy vector not set!')

        if self.sigma_HI is not None:
            return self.sigma_HI
        else:
            # compute cross_section for each energy E in energy_vector
            sigma_HI = [self.physics_ionisation_cross_section_hydrogen(E) for E in self.energy_vector]
            # convert lists to numpy array
            self.sigma_HI = np.asarray(sigma_HI)

            return self.sigma_HI


    def get_photo_ionisation_cross_section_helium1(self):
        """ Returns the photo ionisation cross section for helium1
        if the energy_vector is not None.

        Units:  Takes in energy_vector with enegies in eV
                Return photo_ionisation_cross_section in cm^2
        """
        if self.energy_vector is None:
            raise Exception('Energy vector not set!')

        if self.sigma_HeI is not None:
            return self.sigma_HeI
        else:
            # compute cross_section for each energy E in energy_vector
            sigma_HeI = [self.physics_ionisation_cross_section_helium1(E) for E in self.energy_vector]
            # convert lists to numpy array
            self.sigma_HeI = np.asarray(sigma_HeI)

            return self.sigma_HeI


    def get_photo_ionisation_cross_section_helium2(self):
        """ Returns the photo ionisation cross section for helum2
        if the energy_vector is not None.

        Units:  Takes in energy_vector with enegies in eV
                Return photo_ionisation_cross_section in cm^2
        """
        if self.energy_vector is None:
            raise Exception('Energy vector not set!')

        if self.sigma_HeII is not None:
            return self.sigma_HeII
        else:
            # compute cross_section for each energy E in energy_vector
            sigma_HeII = [self.physics_ionisation_cross_section_helium2(E) for E in self.energy_vector]
            # convert lists to numpy array
            self.sigma_HeII = np.asarray(sigma_HeII)

            return self.sigma_HeII


    def get_tau(self, sim_data, E, index):
        """
        Provided the simulation grid data, a given photon energy E, and
        the index for the computing arrays, compute the optical depth tau
        Ref: equation B13
        """
        # [TODO]: this method can be reimplemented as per the class style.
        # works for now.
        sigma_HI = self.physics_ionisation_cross_section_hydrogen(E)
        sigma_HeI = self.physics_ionisation_cross_section_helium1(E)
        sigma_HeII = self.physics_ionisation_cross_section_helium2(E)

        # [delta r] = kpc, [number density] = cm^-3, [sigma] = cm^2
        tau_HI = sigma_HI * sim_data.delta_radius * np.sum(sim_data.n_H_I[0:index]) * CONSTANT_kpc_to_cm
        tau_HeI = sigma_HeI * sim_data.delta_radius * np.sum(sim_data.n_He_I[0:index]) * CONSTANT_kpc_to_cm
        tau_HeII = sigma_HeII * sim_data.delta_radius * np.sum(sim_data.n_He_II[0:index]) * CONSTANT_kpc_to_cm

        return tau_HI + tau_HeI + tau_HeII


    def get_ionisation_rate_integral_hydrogen(self):
        """ Solves the integral in the equation for the computation of
        ionisation rate of Hydrogen, given the energy vector and the flux vector.
        Note: This function just returns the integral and needs to multiplied
        with beta_1 and n_e to complete the equation.
        Ref: equation (A.6) in [2]
        Units:
        """
        # sanity checks
        if self.energy_vector is None:
            raise Exception('Energy vector not set!')
        if self.flux_vector is None:
            raise Exception('Flux vector not set!')
        if self.sigma_HI is None:
            self.sigma_HI = self.get_photo_ionisation_cross_section_hydrogen()

        if self.integral_hydrogen_ionisation_rate is not None:
            return self.integral_hydrogen_ionisation_rate
        else:
            # get the training_data_size
            train_set_size = self.flux_vector.shape[0]
            # compute the value of integrand for each energy E in energy_vector
            integrand = self.sigma_HI[None, :] * self.flux_vector / self.energy_vector[None, :]
            # placeholder array to store the computed the integrals
            self.integral_hydrogen_ionisation_rate = np.zeros(train_set_size)
            # compute the integrals using simpsons integration
            for i in range(train_set_size):
                self.integral_hydrogen_ionintegral_helium1_ionisation_rateisation_rate[i] = utils_simpson_integration(integrand[i], self.energy_vector)

            return self.integral_hydrogen_ionisation_rate


    def get_ionisation_rate_integral_helium1(self):
        """ Solves the integral in the equation for the computation of
        ionisation rate of helium1, given the energy vector and the flux vector.
        Ref: equation (A.7) in [2]
        Units:
        """
        # sanity checks
        if self.energy_vector is None:
            raise Exception('Energy vector not set!')
        if self.flux_vector is None:
            raise Exception('Flux vector not set!')
        if self.sigma_HeI is None:
            self.sigma_HeI = self.get_photo_ionisation_cross_section_helium1()

        if self.integral_helium1_ionisation_rate is not None:
            return self.integral_helium1_ionisation_rate
        else:
            # get the training_data_size
            train_set_size = self.flux_vector.shape[0]
            # compute the value of integrand for each energy E in energy_vector
            integrand = self.sigma_HeI[None, :] * self.flux_vector / self.energy_vector[None, :]
            # placeholder array to store the computed the integrals
            self.integral_helium1_ionisation_rate = np.zeros(train_set_size)
            # compute the integrals using simpsons integration
            for i in range(train_set_size):
                self.integral_helium1_ionisation_rate[i] = utils_simpson_integration(integrand[i], self.energy_vector)

            return self.integral_helium1_ionisation_rate


    def get_ionisation_rate_integral_helium2(self):
        """ Solves the integral in the equation for the computation of
        ionisation rate of helium2, given the energy vector and the flux vector.
        Ref: equation (A.8) in [2]
        Units:
        """
        # sanity checks
        if self.energy_vector is None:
            raise Exception('Energy vector not set!')
        if self.flux_vector is None:
            raise Exception('Flux vector not set!')
        if self.sigma_HeII is None:
            self.sigma_HeII = self.get_photo_ionisation_cross_section_helium2()

        if self.integral_helium2_ionisation_rate is not None:
            return self.integral_helium2_ionisation_rate
        else:
            # get the training_data_size
            train_set_size = self.flux_vector.shape[0]
            # compute the value of integrand for each energy E in energy_vector
            integrand = self.sigma_HeII[None, :] * self.flux_vector / self.energy_vector[None, :]
            # placeholder array to store the computed the integrals
            self.integral_helium2_ionisation_rate = np.zeros(train_set_size)
            # compute the integrals using simpsons integration
            for i in range(train_set_size):
                self.integral_helium2_ionisation_rate[i] = utils_simpson_integration(integrand[i], self.energy_vector)

            return self.integral_helium2_ionisation_rate


    def reset_pre_computed_variables(self):
        """ This function sets all the pre-computed stored variables to None
        when it is called. Must be called anytime, the data for this class changes.
        """
        # photo ionisation cross sections
        self.sigma_HI = None
        self.sigma_HeI = None
        self.sigma_HeII = None
        # integrals for computing ionisation rates
        self.integral_hydrogen_ionisation_rate = None
        self.integral_helium1_ionisation_rate = None
        self.integral_helium2_ionisation_rate = None


    def set_energy_vector(self, energy_vector):
        """ Updates the energy_vector of shape (n_energy_bins) for the instance of
        this class and resets all the pre-computed variables.
        Units: Takes in energy vector in eV
        """
        self.energy_vector = energy_vector
        self.reset_pre_computed_variables()


    def set_flux_vector(self, flux_vector):
        """ Updates the flux_vector of shape (train_set_size, n_energy_bins) for the
        instance of this class and resets all the pre-computed variables.
        Units:
        """
        self.flux_vector = flux_vector
        self.reset_pre_computed_variables()


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
