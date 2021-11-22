"""
A module to store all physics constants
"""

# Unit conversions
MPC_to_CM = 3.086e24                        # Mega parsec to cm conversion
KPC_to_CM = 3.086e21                        # kilo parsec to cm conversion
KM_to_CM = 1.0e5                            # Kilo meters to cm conversion
ERG_TO_EV = 6.242e+11

# Speed of light
CONSTANT_LIGHT_SPEED = 2.9979e10            # speed of light in cm/second

# Boltzmann's constant
CONSTANT_BOLTZMANN_ERG = 1.3807e-16         # Boltzmann Constant [erg/kelvin]
CONSTANT_BOLTZMANN_EV = 8.617e-05           # Boltzmann Constant [eV/kelvin]

# Hubble's constant
CONSTANT_HUBBLE_Z0 = 67.74                  # Hubble Constant [km /sec.Mpc] at z = 0

# Mass of electron
CONSTANT_MASS_ELECTRON = 9.10938188e-28     # Mass electron (grams)
CONSTANT_MASS_ELECTRON_EV = 511e6           # Mass electron (eV)

# Cosmological parameters
CONSTANT_COSMOS_OMEGA_M = 0.3089            # Total matter density
CONSTANT_COSMO_OMEGA_L = 0.6911             # Dark energy / cosmological constant
CONSTANT_COSMO_OMEGA_B = 0.0486             # Baryon density parameter
CONSTANT_COSMO_H_0 = 67.74                  # Hubble parameter [km/sec/Mpc] at z = 0
CONSTANT_COSMO_H100 = 0.6774                # Hubble parameter / 100
CONSTANT_COSMO_SIGMA8 = 0.8159              # Power spectrum normalization
CONSTANT_COSMO_TAU_THOM = 0.066             # Optical depth of Thomson scattering (not used in calculations right now)
CONSTANT_COSMO_T_CMB_0 = 2.731              # CMB temperature at z = 0 in Kelvin

# Planck's constant
CONSTANT_PLANCK = 6.265e-27                 # PLANCK's Constant [erg sec]
CONSTANT_PLANCK_EV = 4.14e-15               # PLANCK's Constant [eV sec]

# Thompson cross section (σT)
CONSTANT_THOMSON_ELEC_CROSS = 6.6524e-25    # [cm^2]

# Redshift at which gas and CMB temperature were in equilibrium
CONSTANT_z_T_kin_EQ_T_CMB = 250

# number densities at redshift z = 0
CONSTANT_n_H_0 = 1.9e-7                     # H_I + H_II number density in cm^-3
CONSTANT_n_He_0 = 1.5e-8                    # He_I + He_II + He_III number density in cm^-3

# Ionisation edges
IONIZATION_ENERGY_HYDROGEN = 13.6057        # unit is eV
IONIZATION_ENERGY_HELIUM1 = 24.5874         # unit is eV
IONIZATION_ENERGY_HELIUM2 = 54.4228         # unit is eV
