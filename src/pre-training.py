import argparse
import os
import signal

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as torch_data

from common.settings_ode import ode_parameter_limits as ps_ode
from common.settings_sed import p8_limits as ps_sed
from common.settings_sed import SED_ENERGY_MIN, SED_ENERGY_MAX, SED_ENERGY_DELTA
from common.utils import *
from common.physics import *
from common.settings_crt import *
from common.settings import *
from common.data_log import *
from sed import sed_numba
from models_pretraining import *
from random import random

# check for CUDA
if torch.cuda.is_available():
    cuda = True
    device = torch.device("cuda")
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    cuda = False
    device = torch.device("cpu")
    torch.set_default_tensor_type(torch.FloatTensor)


# -----------------------------------------------------------------
#  Main
# -----------------------------------------------------------------
def main(config):

    # -----------------------------------------------------------------
    # create unique output path and run directories, save config
    # -----------------------------------------------------------------
    if config.run == '':
        run_id = 'run_' + utils_get_current_timestamp()
    else:
        run_id = 'run_' + config.run
    config.out_dir = os.path.join(config.out_dir, run_id)
    setattr(config, 'run', run_id)

    utils_create_output_dirs([config.out_dir])
    utils_save_config_to_log(config)
    utils_save_config_to_file(config)

    # # path to store the data and plots after training
    # data_products_path = os.path.join(config.out_dir, DATA_PRODUCTS_DIR)
    # plot_path = os.path.join(config.out_dir, PLOT_DIR)

    # -----------------------------------------------------------------
    # load the data and update config with the dataset conifguration
    # -----------------------------------------------------------------
    parameters, energies, intensities, density_vector, tau,\
     flux_vectors = utils_load_pretraining_data(config.data_dir)
    setattr(config, 'len_SED_input', flux_vectors.shape[1])
    setattr(config, 'n_samples', flux_vectors.shape[0])


    # -----------------------------------------------------------------
    # shuffle/log space
    # -----------------------------------------------------------------
    if PRETRAINING_SHUFFLE:
        np.random.seed(PRETRAINING_SEED)
        indices = np.arange(config.n_samples, dtype=np.int32)
        indices = np.random.permutation(indices)
        flux_vectors = flux_vectors[indices]

    if PRETRAINING_LOG_PROFILES:
        # add a small number to avoid trouble
        flux_vectors = np.log10(flux_vectors + 1.0e-6)

    # -----------------------------------------------------------------
    # convert data into tensors and split it into requried legths
    # -----------------------------------------------------------------

    # numpy array to tensors
    flux_vectors = torch.Tensor(flux_vectors)

    # calculate length for train. val and test dataset from fractions
    train_length = int(PRETRAINING_SPLIT_FRACTION[0] * config.n_samples)
    validation_length = int(PRETRAINING_SPLIT_FRACTION[1] * config.n_samples)
    test_length = config.n_samples - train_length - validation_length

    # split the dataset
    train_dataset, validation_dataset, test_dataset = \
     torch.utils.data.random_split(flux_vectors,
                    (train_length, validation_length,test_length),
                    generator=torch.Generator().manual_seed(PRETRAINING_SEED))

    # -----------------------------------------------------------------
    # dataloaders from dataset
    # -----------------------------------------------------------------
    train_loader = torch_data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
    val_loader = torch_data.DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False)
    train_loader = torch_data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # -----------------------------------------------------------------
    # tensorboard (to check results, visit localhost:6006)
    # -----------------------------------------------------------------
    # data_log = DataLog.getInstance(config.out_dir)
    # data_log.start_server()

    # -----------------------------------------------------------------
    # initialise model
    # -----------------------------------------------------------------
    model = AE1(config)
    print('\n\tusing model AE1 on device: %s\n'%(device))

    if cuda:
        model.cuda()

    # -----------------------------------------------------------------
    # Optimizers
    # -----------------------------------------------------------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        betas=(config.b1, config.b2)
    )


    # -----------------------------------------------------------------
    # book keeping arrays
    # -----------------------------------------------------------------
    avg_train_loss_array = np.empty(0)
    avg_val_loss_mse_array = np.empty(0)

    # -----------------------------------------------------------------
    #  Main training loop
    # -----------------------------------------------------------------
    print("\033[96m\033[1m\nTraining starts now\033[0m")
    for epoch in range(1, config.n_epochs + 1):
        epoch_loss = 0
        # set model mode
        model.train()
        for i, flux_vectors in enumerate(train_loader):
            # train the model here.
            flux_vectors = Variable(flux_vectors)

            # zero the gradients on each iteration
            optimizer.zero_grad()
            regen_flux_vectors = model(flux_vectors)

            # compute loss
            loss = F.mse_loss(input=regen_flux_vectors, target=flux_vectors, reduction='mean')
            loss.backward()
            optimizer.step()

            # sum the loss values
            epoch_loss += loss.item()

        # end-of-epoch book keeping
        train_loss = epoch_loss / len(train_loader)
        avg_train_loss_array = np.append(avg_train_loss_array, train_loss)

        # [TODO] validation

        print("[Epoch %d/%d] [Train loss: %e] [Validation loss: %e][Best_epoch: %d]"
         % (epoch, config.n_epochs, train_loss, 0, 0))


    # -----------------------------------------------------------------
    # Overwrite config object
    # -----------------------------------------------------------------
    utils_save_config_to_log(config)
    utils_save_config_to_file(config)


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='ML-RT2 - pre-training module for SED\'s')

    parser.add_argument('--out_dir', type=str, default='output_pretraining', metavar='(string)',
                        help='Path to output directory, used for all plots and data products, default: ./output_pretraining/')
    parser.add_argument('--data_dir', type=str, default='../data/sed_samples', metavar='(string)',
                        help='Path of the data directory from which data is to be read for training the model, default: ../data/sed_samples')
    parser.add_argument('--run', type=str, default='', metavar='(string)',
                        help='Specific run name for the experiment, default: ./output/timestamp')

    # grid settings
    parser.add_argument("--radius_max", type=float, default=DEFAULT_RADIUS_MAX,
                        help="Maximum radius in kpc. Default = 1500.0")

    parser.add_argument("--radius_min", type=float, default=DEFAULT_RADIUS_MIN,
                        help="Minimum radius in kpc. Default = 0.1")

    parser.add_argument("--delta_radius", type=float, default=DEFAULT_SPATIAL_RESOLUTION,
                        help="Spatial resolution in kpc. Default = 1")

    parser.add_argument("--delta_time", type=float, default=DEFAULT_TEMPORAL_RESOLUTION,
                        help="Temporal resolution in Myr. Default = 0.01")

    # network optimisation
    parser.add_argument("--n_epochs", type=int, default=100,
                        help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="size of the batches (default=32)")

    parser.add_argument("--lr", type=float, default=0.0001,
                        help="adam: learning rate, default=0.0001")
    parser.add_argument("--b1", type=float, default=0.9,
                        help="adam: beta1 - decay of first order momentum of gradient, default=0.9")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: beta2 - decay of first order momentum of gradient, default=0.999")

    # model configuration
    parser.add_argument("--model", type=str, default='AE1', help="Model to use")
    parser.add_argument("--batch_norm", dest='batch_norm', action='store_true',
                        help="use batch normalisation in network (default)")
    parser.add_argument('--no-batch_norm', dest='batch_norm', action='store_false',
                        help="use batch normalisation in network")
    parser.set_defaults(batch_norm=False)

    parser.add_argument("--dropout", dest='dropout', action='store_true',
                        help="use dropout regularisation in network (default)")
    parser.add_argument("--no-dropout", dest='dropout', action='store_false',
                        help="do not use dropout regularisation in network")
    parser.set_defaults(dropout=False)
    parser.add_argument("--dropout_value", type=float, default=0.25, help="dropout probability, default=0.25 ")

    parser.add_argument("--len_latent_vector", type=int, default=8,
                        help="length of reduced SED vector")

    my_config = parser.parse_args()

    my_config.out_dir = os.path.abspath(my_config.out_dir)
    my_config.profile_type = 'C'
    my_config.device = device

    # print summary
    print("\nUsed parameters:\n")
    for arg in vars(my_config):
        print("\t", arg, getattr(my_config, arg))

    # run main program
    main(my_config)
