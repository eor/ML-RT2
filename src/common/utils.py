import numpy as np
import os
import pickle
import torch
import torch.nn.functional as F
import os.path as osp
from datetime import datetime
from configparser import ConfigParser
from numba import jit

# -----------------------------------------------------------------
# join path and check if file exists
# -----------------------------------------------------------------
def utils_join_path(directory, data_file):

    a = osp.join(directory, data_file)

    if not osp.exists(a):
        print('Error: File not found:\n\n  %s\n\nExiting.' % a)
        exit(1)

    return a

# -----------------------------------------------------------------
# create output directories
# -----------------------------------------------------------------
def utils_create_output_dirs(list_of_dirs):

    for x in list_of_dirs:
        if not osp.exists(x):
            os.makedirs(x)
            print('Created directory:\t%s' % x)

# -----------------------------------------------------------------
# Current time stamp as a string
# -----------------------------------------------------------------
def utils_get_current_timestamp():

    # return datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    return datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

# -----------------------------------------------------------------
# Create directories
# -----------------------------------------------------------------
def utils_create_run_directories(main_dir, data_products_dir='data_products', plot_dir='plots'):

    d = osp.join(main_dir, data_products_dir)
    p = osp.join(main_dir, plot_dir)

    print('\nCreating directories:\n')
    print('\t' + main_dir)
    print('\t' + d)
    print('\t' + p)

    os.makedirs(main_dir, exist_ok=False)
    os.makedirs(d, exist_ok=False)
    os.makedirs(p, exist_ok=False)


# -----------------------------------------------------------------
# Write argparse config to ascii file
# -----------------------------------------------------------------
def utils_save_config_to_log(config, file_name='log'):

    p = osp.join(config.out_dir, file_name)

    print('\nWriting config to ascii file:\n')
    print('\t' + p)

    with open(p, 'w') as f:
        for arg in vars(config):
            line = str(arg) + '\t' + str(getattr(config, arg)) + '\n'
            f.write(line)

        time = utils_get_current_timestamp()
        f.write('\ncurrent time stamp\t' + time + '\n')


# -----------------------------------------------------------------
# Write argparse config to binary file (to re-use later)
# -----------------------------------------------------------------
def utils_save_config_to_file(config, file_name='config.dict'):

    p = osp.join(config.out_dir, file_name)

    print('\nWriting config to binary file:\n')
    print('\t' + p)

    with open(p, 'wb') as f:
        pickle.dump(config, f)

# -----------------------------------------------------------------
# Load argparse config from binary file
# -----------------------------------------------------------------
def utils_load_config(path, file_name='config.dict'):

    if path.endswith(file_name):
        p = path
    else:
        p = osp.join(path, file_name)

    print('\nLoading config object from file:\n')
    print('\t' + p)

    with open(p, 'rb') as f:
        config = pickle.load(f)

    return config

# -----------------------------------------------------------------
# Load pre-training dataset
# -----------------------------------------------------------------
def utils_load_pretraining_data(path, file_name='data_pretraining.npy.npz'):
    if path.endswith(file_name):
        p = path
    else:
        p = osp.join(path, file_name)

    print('\nLoading pretraining data from disk.....')
    data = np.load(p)
    print('loaded data arrays:', data.files)

    parameters = data['parameters']
    energies = data['energies']
    intensities = data['intensities']
    density_vector = data['density_vector']
    tau = data['tau']
    flux_vector = data['flux_vector']

    return parameters, energies, intensities, density_vector, tau, flux_vector

# -----------------------------------------------------------------
# save flux_vectors (true & regenerated)
# -----------------------------------------------------------------
def utils_save_pretraining_test_data(flux_vectors_true, flux_vectors_gen, path, epoch, prefix='test'):

    flux_vectors_true_filename = prefix + '_flux_vectors_true_%d_epochs.npy' % (epoch)
    flux_vectors_gen_filename = prefix + '_flux_vectors_gen_%d_epochs.npy' % (epoch)

    flux_vectors_true_path = osp.join(path, flux_vectors_true_filename)
    flux_vectors_gen_path = osp.join(path, flux_vectors_gen_filename)

    print('\nSaving results in the following files:\n')
    print('\t%s' % flux_vectors_true_path)
    print('\t%s\n' % flux_vectors_gen_path)

    np.save(flux_vectors_true_path, flux_vectors_true)
    np.save(flux_vectors_gen_path, flux_vectors_gen)


@jit(nopython=True)
def utils_simpson_integration(y,x):
    # source: https://masonstoecker.com/2021/04/03/Simpson-and-Numba.html
    n = len(y)-1
    h = np.zeros(n)
    for i in range(n):
        h[i] = x[i+1]-x[i]
        if h[i] == 0:
            np.delete(h,i)
            np.delete(y,i)
    n = len(h)-1
    s = 0
    for i in range(1,n,2):
        a = h[i]*h[i]
        b = h[i]*h[i-1]
        c = h[i-1]*h[i-1]
        d = h[i] + h[i-1]
        alpha = (2*a+b-c)/h[i]
        beta  = d*d*d/b
        gamma = (-a+b+2*c)/h[i-1]
        s += alpha*y[i+1]+beta*y[i]+gamma*y[i-1]

    if ((n+1)%2 == 0):
        alpha = h[n-1]*(3-h[n-1]/(h[n-1]+h[n-2]))
        beta = h[n-1]*(3+h[n-1]/h[n-2])
        gamma = -h[n-1]*h[n-1]*h[n-1]/(h[n-2]*(h[n-1]+h[n-2]))
        return (s+alpha*y[n]+beta*y[n-1]+gamma*y[n-2])/6
    else:
        return s/6
