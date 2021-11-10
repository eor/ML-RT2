import argparse
import os
import numpy as np

from common.settings_crt import *
from common.settings_ode import *
from common.settings_sed import *
from common.physics import *


class SimulationData:
    """
    Set up the data structure which contains simulation specific data, most importantly the
    arrays for the quantities whose profiles we want to obtain via the radiative transfer simulations,
    i.e. ionisation fractions and temperatures.

    The class uses a flat density profile for initialising the arrays.
    NFW profile or other density profiles might follow later.
    """

    def __init__(self, conf):

        self.redshift = conf.redshift

        self.lifetime = conf.source_lifetime

        self.current_time = 0.0

        self.radius_max = conf.radius_max
        self.radius_min = conf.radius_min
        self.delta_radius = conf.delta_radius

        self.grid_length = int((self.radius_max - self.radius_min) / self.delta_radius)

        # ionisation fractions
        self.x_H_I = np.ones(self.grid_length)
        self.x_H_II = np.zeros(self.grid_length)

        self.x_He_I = np.ones(self.grid_length)
        self.x_He_II = np.zeros(self.grid_length)
        self.x_He_III = np.zeros(self.grid_length)

        # IGM temperature
        init_T = CONSTANT_T_CMB_0 * np.power(1. + self.redshift, 2) / (1.0 + CONSTANT_z_T_kin_EQ_T_CMB)
        self.T = np.ones(self.grid_length) * init_T

        # over density = 1.0 for now.
        # can be changed later to use custom density profiles
        self.over_densities = np.ones(self.grid_length)

        # number densities (free electrons,  hydrogen, helium)
        self.n_e = np.zeros(self.grid_length)

        init_n_H = CONSTANT_n_H_0 * np.power(1.0 + self.redshift, 3) * self.over_densities
        self.n_H_I = np.ones(self.grid_length) * init_n_H

        init_n_He = CONSTANT_n_He_0 * np.power(1.0 + self.redshift, 3) * self.over_densities
        self.n_He_I = np.ones(self.grid_length) * init_n_He

        # the following might not be needed
        self.n_H_II = np.zeros(self.grid_length)
        self.n_He_II = np.zeros(self.grid_length)
        self.n_He_III = np.zeros(self.grid_length)

        # set boundary conditions, i.e. full ionisation at r=0 at the start of the simulation
        # if conf.radius_min == 0.0:
        #     self.x_H_I[0] = 0.0
        #     self.x_H_II[0] = 1.0
        #     self.x_He_I[0] = 0.0
        #     self.x_He_II[0] = 0.0
        #     self.x_He_III[0] = 1.0

    def get_tau(self, E, index):
        """
        For a given Photon energy E and index for the computing arrays, compute the optical depth tau
        """

        sigma_HI = physics_ionisation_cross_section_hydrogen(E)
        sigma_HeI = physics_ionisation_cross_section_helium1(E)
        sigma_HeII = physics_ionisation_cross_section_helium2(E)

        tau_HI = sigma_HI * self.delta_radius * np.sum(self.n_H_I[0:index+1])
        tau_HeI = sigma_HeI * self.delta_radius * np.sum(self.n_He_I[0:index+1])

        # TODO tau_HeII, figure out the other densities

        return tau_HI + tau_HeI


# -----------------------------------------------------------------
#  Main
# -----------------------------------------------------------------
def main(config):



    # 1. load neural ode model

    # 2. set up run directory

    # 3. init data class

    sim = SimulationData(config)

    tau = sim.get_tau(13.61, 15000-1)

    print(tau)


# -----------------------------------------------------------------
#  Parse user input
# -----------------------------------------------------------------
if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='CRT - Cosmological radiative transfer with PINNs (PINN)')

    # everything I / O related
    parser.add_argument('--out_dir', type=str, default='output', metavar='(string)',
                        help='Path to output directory, used for all plots and data products, default: ./output/')

    parser.add_argument('--model_path', type=str, required=True, metavar='(string)',
                        help='Path to pre-trained neural ODE model')

    # computing grid settings
    parser.add_argument("--radius_max", type=float, default=DEFAULT_RADIUS_MAX,
                        help="Maximum radius in kpc. Default = 1500.0")

    parser.add_argument("--radius_min", type=float, default=DEFAULT_RADIUS_MIN,
                        help="Minimum radius in kpc. Default = 0.1")

    parser.add_argument("--delta_radius", type=float, default=DEFAULT_SPATIAL_RESOLUTION,
                        help="Spatial resolution in kpc. Default = 0.1")

    parser.add_argument("--delta_time", type=float, default=DEFAULT_TEMPORAL_RESOLUTION,
                        help="Temporal resolution in Myr. Default = 0.01")

    # source parameters
    parser.add_argument("--halo_mass_log", type=float, default=10.0,
                        help="Log10 host halo dark matter mass (8-15). Default = 10.0")

    parser.add_argument("--redshift", type=float, default=9.0, help="Source redshift (6-13). Default = 9.0")

    parser.add_argument("--source_lifetime", type=float, default=10.0,
                        help="Source lifetime (0.1-20.0 Myr). Default = 10.0")

    parser.add_argument("--source_qso_alpha", type=float, default=1.0,
                        help="QSO power law index (0.0-2.0). Default = 1.0")

    parser.add_argument("--source_qso_efficiency", type=float, default=0.1,
                        help="QSO efficiency (0.0-1.0). Default = 0.1")

    parser.add_argument("--source_stars_fesc", type=float, default=0.1,
                        help="Escape fraction for stellar UV radiation (0.0-1.0). Default = 0.1")

    parser.add_argument("--source_stars_imf_slope", type=float, default=2.35,
                        help="IMF slope stellar UV radiation (0.0-1.5). Default = 2.35")

    parser.add_argument("--source_stars_min_mass", type=float, default=20.0,
                        help="Minimum stellar mass (5.0-500 M_sol). Default = 20.0")

    my_config = parser.parse_args()

    my_config.out_dir = os.path.abspath(my_config.out_dir)


    print("\n CRT test run")

    print("\nUsed parameters:\n")
    for arg in vars(my_config):
        print("\t", arg, getattr(my_config, arg))

    main(my_config)
