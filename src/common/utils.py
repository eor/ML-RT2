import numpy as np
import os
import pickle
import torch
import torch.nn.functional as F
import os.path as osp
from datetime import datetime
from configparser import ConfigParser


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
