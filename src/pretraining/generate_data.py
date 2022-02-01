#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math as m
import tqdm
import multiprocessing
import os
from pyDOE import lhs
from timeit import default_timer as timer

import sys; sys.path.append('..')

import common.sed_numba as sed_nb
from common.settings_sed import SED_ENERGY_MIN, SED_ENERGY_MAX
from common.settings_sed import p8_limits as parameter_ranges
from common.physics import *
from common.settings import DATA_GENERATION_SEED
from common.physics_constants import *

from pretraining.settings import tau_input_vector_limits


# -----------------------------------------------------------------
# obtain sample set, normalised to [0, 1] range
# -----------------------------------------------------------------
def get_normalised_sample_set(n_parameters, n_samples):
    """
    This function returns n_samples for a given number of parameters (n_parameters).
    It uses latin hypercube sampling via the pyDOE package and therefore all samples are normalised to [0, 1].
    """

    return lhs(n=n_parameters, samples=n_samples)


# -----------------------------------------------------------------
#  adjust sample ranges for all parameters
# -----------------------------------------------------------------
def adjust_sample_to_parameter_ranges(p_list, samples):
    """
    This function re-casts normalised parameter sample values from the [0,1] interval
    to their respective parameter interval [a,b].

    Input: The normalised parameter sample set (N samples, M parameters)
           A list of parameter intervals (M x [a,b])
    """

    if len(samples[0]) != len(p_list):
        print('Error sample set and parameter dictionary incompatible. Exiting.')
        exit(1)

    N = samples.shape[0]
    M = len(p_list)

    for i in range(0, N):
        for j in range(0, M):
            x = samples[i, j]
            limits = p_list[j]
            a = limits[0]
            b = limits[1]
            samples[i, j] = recast_sample_value(x, a, b)

    return samples


# -----------------------------------------------------------------
#  adjust sample ranges for a single parameter
# -----------------------------------------------------------------
def recast_sample_value(x, a, b):
    """
    Given a float x of interval [0,1] return the corresponding value for the interval [a,b].
    """
    return a + (b - a) * x


# -----------------------------------------------------------------
# set up a directory for our data set
# -----------------------------------------------------------------
def setup_sample_dir(path, key, nSamples):
    directory = '%s/sed_samples_%s_N%d' % (path, key, nSamples)

    if os.path.exists(directory):

        # if dir exists, rename it
        from time import strftime, gmtime
        timestamp = strftime("%Y%m%d_%H:%M:%S", gmtime())

        directoryBackup = directory + '_backup_' + timestamp

        # print directory, directoryBackup

        if not os.path.exists(directoryBackup):
            os.rename(directory, directoryBackup)
        else:
            print('Error: Backup directory \'%s\' already exists' % directory)
            exit(1)

        os.makedirs(directory)
        print('Renamed ' + directory + ' to ' + directoryBackup)

    else:
        os.makedirs(directory)
        print('Created empty directory ' + directory)

    return directory


# -----------------------------------------------------------------
# write samples to file
# -----------------------------------------------------------------
def write_data(target_file, parameters, energies, intensities,
               tau_input_vector, tau, flux_vector, directory=None):

    if directory:
        path = directory + '/' + target_file
    else:
        path = './' + target_file

    np.savez_compressed(path,
                        parameters=parameters,
                        energies=energies,
                        intensities=intensities,
                        tau_input_vector=tau_input_vector,
                        tau=tau,
                        flux_vector=flux_vector)


# -----------------------------------------------------------------
# generate pretraining data
# -----------------------------------------------------------------
def generate_data(parameters, tau_per_sed=10):

    # generate sed from parameters (returns arrays for photon energies and intensities)
    energies, intensities = sed_nb.generate_SED_IMF_PL(halo_mass=parameters[0],
                                                       redshift=parameters[1],
                                                       eLow=SED_ENERGY_MIN,
                                                       eHigh=SED_ENERGY_MAX,
                                                       N=2000,
                                                       logGrid=True,
                                                       starMassMin=parameters[7],
                                                       starMassMax=500,
                                                       imfBins=50,
                                                       imfIndex=parameters[6],
                                                       fEsc=parameters[5],
                                                       alpha=parameters[3],
                                                       qsoEfficiency=parameters[4],
                                                       targetSourceAge=parameters[2])

    # sample parameters for computing tau
    r = np.random.randint(tau_input_vector_limits[0][0], tau_input_vector_limits[0][1], size=(tau_per_sed, 1))
    # redshift used to obtain I(E) from source.
    redshift = parameters[1] * np.ones((tau_per_sed, 1))

    # compute total initial densities before ionisation using redshift values
    n_H_0 = CONSTANT_n_H_0 * np.power(1 + redshift, 3)
    n_He_0 = CONSTANT_n_He_0 * np.power(1 + redshift, 3)

    # sample random ionisation fractions between 0 and 1 for neutral hydrogen,
    # helium and singly ionised helium.
    ionisation_fraction_H_I = np.random.random(size=(tau_per_sed, 1))
    ionisation_fraction_He_I = np.random.random(size=(tau_per_sed, 1))
    ionisation_fraction_He_II = 1 - ionisation_fraction_He_I

    # use the sampled ionisation fractions and initial number densities to compute
    # number densities of neutral hydrogen, helium and singly ionised helium.
    num_density_H_I = n_H_0 * ionisation_fraction_H_I
    num_density_He_I = n_He_0 * ionisation_fraction_He_I
    num_density_He_II = n_He_0 * ionisation_fraction_He_II

    # concatenate individual parameters to tau_input_vector
    tau_input_vector = np.concatenate((r, redshift, num_density_H_I, num_density_He_I, num_density_He_II), axis=1)

    # obtain arrays of photo-ionisation cross-sections corresponding to the photon energies array
    physics = Physics.getInstance()
    physics.set_energy_vector(energies)
    sigmas_H_I = physics.get_photo_ionisation_cross_section_hydrogen()
    sigmas_He_I = physics.get_photo_ionisation_cross_section_helium1()
    sigmas_He_II = physics.get_photo_ionisation_cross_section_helium2()

    # generate tau from tau_input_vector
    tau = sigmas_H_I[np.newaxis, :] * num_density_H_I
    tau += sigmas_He_I[np.newaxis, :] * num_density_He_I
    tau += sigmas_He_II[np.newaxis, :] * num_density_He_II
    tau *= r * KPC_to_CM

    # generate flux_vector (add small number to r to avoid division by zero)
    flux_vector = (intensities[np.newaxis, :] * np.exp(-1 * tau)) / (4 * np.pi * np.power(r + 1e-5, 2))

    # reshape/broadcast input parameters to shape (tau_per_sed, parameters)
    parameters = np.repeat(parameters[np.newaxis, :], tau_per_sed, axis=0)
    energies = np.repeat(energies[np.newaxis, :], tau_per_sed, axis=0)
    intensities = np.repeat(intensities[np.newaxis, :], tau_per_sed, axis=0)

    return parameters, energies, intensities, tau_input_vector, tau, flux_vector


# -----------------------------------------------------------------
# main
# -----------------------------------------------------------------
def main(path, key, n_samples):

    # set up file name and directory
    sample_file = 'sed_%s_N%d.npy' % (key, n_samples)
    sample_dir = setup_sample_dir(path, key, n_samples)

    # generate a sample parameter set
    sample_set = get_normalised_sample_set(n_parameters=len(parameter_ranges), n_samples=n_samples)
    sample_set = adjust_sample_to_parameter_ranges(p_list=parameter_ranges, samples=sample_set)

    # using multiprocessing and the sampled parameters, generate data
    with multiprocessing.Pool() as pool:
        (parameters,
         energies,
         intensities,
         tau_input_vector,
         tau,
         flux_vector) = zip(*tqdm.tqdm(pool.imap(generate_data, sample_set), total=sample_set.shape[0]))

    # concatenate numpy arrays
    parameters = np.concatenate(parameters, axis=0)
    energies = np.concatenate(energies, axis=0)
    intensities = np.concatenate(intensities, axis=0)
    tau_input_vector = np.concatenate(tau_input_vector, axis=0)
    tau = np.concatenate(tau, axis=0)
    flux_vector = np.concatenate(flux_vector, axis=0)

    # write the data
    write_data(target_file=sample_file,
               parameters=parameters,
               energies=energies,
               intensities=intensities,
               tau_input_vector=tau_input_vector,
               tau=tau,
               flux_vector=flux_vector,
               directory=sample_dir)


# -----------------------------------------------------------------
# execute this when file is executed
# -----------------------------------------------------------------
if __name__ == "__main__":

    np.random.seed(DATA_GENERATION_SEED)

    sample_directory = '../../data/pretraining/'

    start = timer()
    # main(path=sample_directory, key='set_1', n_samples=10000)
    main(path=sample_directory, key='set_1', n_samples=1000)
    end = timer()
    print("total time:", (end - start))
