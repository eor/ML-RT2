#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('../sed')
sys.path.append('../common')
import os
import math as m
import numpy as np
from pyDOE import lhs


from settings_sed import p8_limits as ps_sed
from settings_sed import SED_ENERGY_MIN, SED_ENERGY_MAX, SED_ENERGY_DELTA


# we need
# 1. function to do the latin hypercube sampling
# 2. function to run the SED generator & return the final SED vector
# 3. a function to collect the vectors, build a .npy file and save it to ../data

# -----------------------------------------------------------------
# obtain latin hypercube sample, normalised to [0, 1] range
# -----------------------------------------------------------------
def get_norm_sample_set(n_parameters, n_samples):
    return lhs(n=n_parameters, samples=n_samples)


# -----------------------------------------------------------------
#  adjust sample ranges for all parameters
# -----------------------------------------------------------------
def adjust_sample_to_parameter_ranges(p_list, samples):
    """
    Input: Parameter array of M parameters and their ranges, the sample set (NxM)

    Sample values are [0,1] and will be re-cast to fall into their respective parameter interval [a,b]
    """

    if len(samples[0]) != len(p_list):
        print('Error sample set and parameter dectionary incompatible. Exiting.')
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
    given a float x of interval [0,1] return the corresponding value for the interval [a,b]
    """
    return a + (b - a) * x


# -----------------------------------------------------------------
# set up a directory for our samples
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


#-----------------------------------------------------------------
# write samples to file
#-----------------------------------------------------------------
def write_data(sample_data, target_file, directory=None):

    if directory:
        path = directory+'/' + target_file
    else:
        path = './' + target_file


    # TODO: write npy file


# -----------------------------------------------------------------
# main
# -----------------------------------------------------------------
def create_sample_main(path, key, n_samples):

    sample_file = 'sed_%s_N%d.npy' % (key, n_samples)

    sample_dir = setup_sample_dir(path, key, n_samples)

    p_list = ps_sed
    sample_set = get_norm_sample_set(n_parameters=len(p_list), n_samples=n_samples)
    sample_set = adjust_sample_to_parameter_ranges(p_list=p_list, samples=sample_set)

    sample_data = []

    for sample in sample_set:
        print("Generating SED for parameter sample ... ")
        print(sample)
        # TODO: run sed gen here, add results to a 2d array called sample_data

    # print sample_set
    # write_data(sample_data=sample_data, target_file=sample_file, directory=sample_dir)



# -----------------------------------------------------------------
# execute this when file is executed
# -----------------------------------------------------------------
if __name__ == "__main__":

    dir = '../../data/sed_samples/'

    create_sample_main(path=dir, key='set_1', n_samples=100)


