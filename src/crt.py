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
        # source parameters for sed_generation
        self.halo_mass_log = conf.halo_mass_log
        self.redshift = conf.redshift
        self.lifetime = conf.source_lifetime
        self.source_qso_alpha = conf.source_qso_alpha
        self.source_qso_efficiency = conf.source_qso_efficiency
        self.source_stars_fesc = conf.source_stars_fesc
        self.source_stars_imf_slope = conf.source_stars_imf_slope
        self.source_stars_min_mass = conf.source_stars_min_mass

        self.current_time = 0.0

        self.radius_max = conf.radius_max
        self.radius_min = conf.radius_min
        self.delta_radius = conf.delta_radius

        self.delta_time = conf.delta_time

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

        # number density arrays for total hydrogen and helium in units of cm^-3
        # these should be static but static class variables are a headache in Python
        init_n_hydrogen = CONSTANT_n_H_0 * np.power(1.0 + self.redshift, 3) * self.over_densities
        self.n_hydrogen = np.ones(self.grid_length) * init_n_hydrogen

        init_n_helium = CONSTANT_n_He_0 * np.power(1.0 + self.redshift, 3) * self.over_densities
        self.n_helium = np.ones(self.grid_length) * init_n_helium

        # number densities of free electrons in units of cm^-3, initially zero
        self.n_e = np.zeros(self.grid_length)

        # number density arrays in units of cm^-3
        # they will be updated using the total number densities and the ionisation fraction arrays
        self.n_H_I = np.ones(self.grid_length) * init_n_hydrogen
        self.n_H_II = np.zeros(self.grid_length)
        self.n_He_I = np.ones(self.grid_length) * init_n_helium
        self.n_He_II = np.zeros(self.grid_length)
        self.n_He_III = np.zeros(self.grid_length)

    def update_arrays(self, radial_index):
        """
        Following the ODE step this function updates the number density and ionisation fraction arrays, i.e.
        n_e, n_H_I, n_H_II, n_He_I, n_He_II, n_He_II, x_H_I, x_He_I using the quantities the ODE solver returns, i.e.
        the ionisation fractions (x_H_II, x_He_II, x_He_III), and the overall density arrays n_hydrogen, n_helium.
        """

        # update ionisation fractions
        self.x_H_I[radial_index] = 1.0 - self.x_H_II[radial_index]
        self.x_He_I[radial_index] = 1.0 - self.x_He_II[radial_index] - self.x_He_III[radial_index]

        # update number densities
        self.n_H_I[radial_index] = self.n_hydrogen[radial_index] * self.x_H_I[radial_index]
        self.n_H_II[radial_index] = self.n_hydrogen[radial_index] * self.x_H_II[radial_index]

        self.n_He_I[radial_index] = self.n_helium[radial_index] * self.x_He_I[radial_index]
        self.n_He_II[radial_index] = self.n_helium[radial_index] * self.x_He_II[radial_index]
        self.n_He_III[radial_index] = self.n_helium[radial_index] * self.x_He_III[radial_index]

        # electron number density = sum of number densities of ionised H, He and doubly ionised He
        self.n_e[radial_index] = self.n_H_II[radial_index] + self.n_He_II[radial_index] + 2*self.n_He_III[radial_index]

    def update_current_time(self):
        """
        The function updates, i.e. increments, the simulation's current time state by delta time.
        """
        self.current_time += self.delta_radius


# -----------------------------------------------------------------
#  Main
# -----------------------------------------------------------------
def main(config):

    # 1. load neural ode model

    # 2. set up run directory

    # 3. init data class
    sim = SimulationData(config)

    # 4. get SED

    # 5. run simulation
    while sim.current_time < sim.lifetime:

        print("Current time: %3f Myr" % sim.current_time)
        for radial_index in range(0, sim.grid_length):

            print("Current radius r=%.3f kpc" % (radial_index * sim.delta_radius))

            for energy in np.arange(13.01, 100, 0.1):

                physics_tau(sim, energy, radial_index)

            # load NN inputs: N(E) and state vector (tau, x_i, T, time)
            # tau
            # solve ode
            # sanity checks / regularisation to catch possible mistakes, e.g. ionisation fractions >1 or <0.
            # update sim arrays
            sim.update_arrays(radial_index)

        # all done with the grid for this time step
        sim.update_current_time()

    # 6. write data
    # 7. analysis or plots

# -----------------------------------------------------------------
#  Parse user input
# -----------------------------------------------------------------
if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='CRT - Cosmological radiative transfer with PINNs (PINN)')

    # everything I / O related
    parser.add_argument('--out_dir', type=str, default='output', metavar='(string)',
                        help='Path to output directory, used for all plots and data products, default: ./output/')

    # parser.add_argument('--model_path', type=str, required=True, metavar='(string)',
    #                     help='Path to pre-trained neural ODE model')

    # computing grid settings
    parser.add_argument("--radius_max", type=float, default=DEFAULT_RADIUS_MAX,
                        help="Maximum radius in kpc. Default = 1500.0")

    parser.add_argument("--radius_min", type=float, default=DEFAULT_RADIUS_MIN,
                        help="Minimum radius in kpc. Default = 0.1")

    parser.add_argument("--delta_radius", type=float, default=DEFAULT_SPATIAL_RESOLUTION,
                        help="Spatial resolution in kpc. Default = 1")

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
