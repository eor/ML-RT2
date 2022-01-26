import torch
import pickle
import os
import numpy as np
from numba import jit
from configparser import ConfigParser
from datetime import datetime
import os.path as osp
import torch.nn.functional as F

import sys; sys.path.append('..')

from pretraining.models import *
from common.settings import *


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

    parameters = data['parameters']
    energies = data['energies']
    intensities = data['intensities']
    tau_input_vector = data['tau_input_vector']
    tau = data['tau']
    flux_vector = data['flux_vector']
    print('loaded data arrays:', data.files)

    return parameters, energies, intensities, tau_input_vector, tau, flux_vector


# -----------------------------------------------------------------
# save state of pretraining model
# -----------------------------------------------------------------
def utils_load_pretraining_model(run_dir_path, best_model=False, file_name=None):
    # initialise the model
    config = utils_load_config(run_dir_path)
    if config.model == 'AE1':
        model = AE1(config)

    # if no file name is provided, construct one here
    if file_name is None:
        if best_model:
            file_name = 'best_model_pretraining_%d_epochs.pth.tar' % (config.best_epoch)
        else:
            file_name = 'model_pretraining_%d_epochs.pth.tar' % (config.n_epochs)

    # load the saved parameters into the model
    model_path = osp.join(run_dir_path, DATA_PRODUCTS_DIR, file_name)
    model.load_state_dict(torch.load(model_path))
    print('\nLoaded pre-trained model from:\t%s\n' % model_path)

    # switch the model to eval mode
    model.eval()

    return model


# -----------------------------------------------------------------
# save state of pretraining model
# -----------------------------------------------------------------
def utils_save_pretraining_model(state, path, n_epoch, best_model=False, file_name=None):

    # if no file name is provided, construct one here
    if file_name is None:
        file_name = 'model_pretraining_%d_epochs.pth.tar' % (n_epoch)

        if best_model:
            file_name = 'best_' + file_name

    path = osp.join(path, file_name)
    torch.save(state, path)
    print('Saved model to:\t%s' % path)


# -----------------------------------------------------------------
# save pretraining loss as numpy object
# -----------------------------------------------------------------
def utils_save_loss(loss_array, path, n_epoch, prefix='train'):

    file_name = prefix + '_pretraining_loss_%d_epochs.npy' % (n_epoch)
    path = osp.join(path, file_name)
    np.save(path, loss_array)
    print('Saved %s loss function to:\t%s' % (prefix, path))


# -----------------------------------------------------------------
# save flux_vectors (true & regenerated)
# -----------------------------------------------------------------
def utils_save_pretraining_test_data(flux_vectors_true, flux_vectors_gen, parameters, path, epoch, prefix='test'):

    parameters_filename = prefix + '_parameters_%d_epochs.npy' % (epoch)
    flux_vectors_true_filename = prefix + '_flux_vectors_true_%d_epochs.npy' % (epoch)
    flux_vectors_gen_filename = prefix + '_flux_vectors_gen_%d_epochs.npy' % (epoch)

    parameters_path = osp.join(path, parameters_filename)
    flux_vectors_true_path = osp.join(path, flux_vectors_true_filename)
    flux_vectors_gen_path = osp.join(path, flux_vectors_gen_filename)

    print('\nSaving results in the following files:\n')
    print('\t%s' % flux_vectors_true_path)
    print('\t%s' % flux_vectors_gen_path)
    print('\t%s\n' % parameters_path)

    np.save(flux_vectors_true_path, flux_vectors_true)
    np.save(flux_vectors_gen_path, flux_vectors_gen)
    np.save(parameters_path, parameters)


@jit(nopython=True)
def utils_simpson_integration(y, x):

    # source: https://masonstoecker.com/2021/04/03/Simpson-and-Numba.html

    n = len(y) - 1
    h = np.zeros(n)
    for i in range(n):
        h[i] = x[i + 1] - x[i]
        if h[i] == 0:
            np.delete(h, i)
            np.delete(y, i)
    n = len(h) - 1
    s = 0
    for i in range(1, n, 2):
        a = h[i] * h[i]
        b = h[i] * h[i - 1]
        c = h[i - 1] * h[i - 1]
        d = h[i] + h[i - 1]
        alpha = (2 * a + b - c) / h[i]
        beta = d * d * d / b
        gamma = (-a + b + 2 * c) / h[i - 1]
        s += alpha * y[i + 1] + beta * y[i] + gamma * y[i - 1]

    if (n + 1) % 2 == 0:
        alpha = h[n - 1] * (3 - h[n - 1] / (h[n - 1] + h[n - 2]))
        beta = h[n - 1] * (3 + h[n - 1] / h[n - 2])
        gamma = -h[n - 1] * h[n - 1] * h[n - 1] / (h[n - 2] * (h[n - 1] + h[n - 2]))
        return (s + alpha * y[n] + beta * y[n - 1] + gamma * y[n - 2]) / 6
    else:
        return s / 6
