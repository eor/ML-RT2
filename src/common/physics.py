import numpy as np


CONSTANT_T_CMB_0 = 2.731                    # CMB temperature at redshift z=0 in Kelvin
CONSTANT_z_T_kin_EQ_T_CMB = 250             # redshift at which gas and CMB temperature were in equilibrium


CONSTANT_n_H_0 = 1.9e-7                     # H_I + H_II number density at z=0 in cm^-3
CONSTANT_n_He_0 = 1.5e-8                    # He_I + He_II + He_III number density at z=0 in cm^-3

IONIZATION_ENERGY_HYDROGEN = 13.6057        # unit is eV
IONIZATION_ENERGY_HELIUM1 = 24.5874         # unit is eV
IONIZATION_ENERGY_HELIUM2 = 54.4228         # unit is eV


def physics_ionisation_cross_section_hydrogen(photon_energy):
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


def physics_ionisation_cross_section_helium1(photon_energy):
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


def physics_ionisation_cross_section_helium2(photon_energy):
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
