import argparse
import os
import signal
import sys
import torch
import numpy as np
from random import random
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import sys; sys.path.append('..')

import common.sed_numba as sed_numba
from common.settings_ode import ode_parameter_limits as ps_ode
from common.settings_sed import p8_limits as ps_sed
from common.settings_sed import SED_ENERGY_MIN, SED_ENERGY_MAX, SED_ENERGY_DELTA
from common.utils import *
from common.physics import *
from common.settings_crt import *
from common.settings import *
from common.data_log import *

from pretraining.settings import tau_input_vector_limits

from ode.models import *
from ode.ode_system import *
from ode.ode_data import *

# check for CUDA
if torch.cuda.is_available():
    cuda = True
    device = torch.device("cuda")
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    cuda = False
    device = torch.device("cpu")
    torch.set_default_tensor_type(torch.FloatTensor)


def force_stop_signal_handler(sig, frame):
    global FORCE_STOP
    if not FORCE_STOP:
        FORCE_STOP = True
        print("\033[96m\033[1m\nTraining will stop after this epoch. Please wait.\033[0m\n")


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

    utils_create_output_dirs([config.out_dir])
    # utils_create_run_directories(config.out_dir, DATA_PRODUCTS_DIR, PLOT_DIR)
    utils_save_config_to_log(config)
    utils_save_config_to_file(config)

    # data_products_path = os.path.join(config.out_dir, DATA_PRODUCTS_DIR)
    # plot_path = os.path.join(config.out_dir, PLOT_DIR)

    # -----------------------------------------------------------------
    # generate data to serve as test set and validation set
    # -----------------------------------------------------------------
    ode_data = ODEData(config.batch_size, device)
    val_loader = ode_data.generate_data(config.val_set_size, mode='val')
    test_loader = ode_data.generate_data(config.test_set_size, mode='test')

    # -----------------------------------------------------------------
    # tensorboard (to check results, visit localhost:6006)
    # -----------------------------------------------------------------
    data_log = DataLog.getInstance(config.out_dir)
    data_log.start_server()

    # initialise the model
    u_approximation = MLP1(config)

    if cuda:
        u_approximation.cuda()

    # initialise the ODE equation
    ode_equation = ODE(config)

    # initialise the Physics class
    physics = Physics.getInstance()

    # -----------------------------------------------------------------
    # Optimizers
    # -----------------------------------------------------------------
    optimizer = torch.optim.Adam(
        u_approximation.parameters(),
        lr=config.lr,
        betas=(config.b1, config.b2)
    )

    # -----------------------------------------------------------------
    # book keeping arrays
    # -----------------------------------------------------------------
    train_loss_array = np.empty(0)
    val_loss_mse_array = np.empty(0)
    val_loss_dtw_array = np.empty(0)

    # -----------------------------------------------------------------
    # FORCE_STOP
    # -----------------------------------------------------------------
    global FORCE_STOP
    FORCE_STOP = False
    if FORCE_STOP_ENABLED:
        signal.signal(signal.SIGINT, force_stop_signal_handler)
        print('\n Press Ctrl + C to stop the training anytime and exit while saving the results.\n')

    print("\033[96m\033[1m\nTraining starts now\033[0m")
    for epoch in range(1, config.n_epochs + 1):

        # generate train data randomly at the start of every epoch.
        train_loader = ode_data.generate_data(config.train_set_size, mode='train')

        # initialise variable to compute average of all
        # train losses over all batches.
        epoch_loss = 0
        epoch_loss_x_H_II = 0
        epoch_loss_T = 0
        epoch_loss_x_He_II = 0
        epoch_loss_x_He_III = 0

        for batch, (x_flux_vector, x_state_vector, x_time_vector, parameter_vectors, energy_vectors, target_residuals) in enumerate(train_loader):

            # TODO: look for boundary conditions???

            # [Issue] generating training data after every epoch leads to lots of noise in loss function
            # possible fixes: generate training data/validation data in a systematic way and
            # then evaluate the model.
            # retrieve train_set_size

            # update the data in this physics module
            physics.set_energy_vector(energy_vectors.cpu().numpy()[0])
            physics.set_flux_vector(x_flux_vector.cpu().numpy())

            # As we need to differenitate input w.r.t to time, set time_vector requires_grad to True
            x_time_vector.requires_grad = True

            # Loss based on CRT ODEs
            r_x_H_II, r_x_He_II, r_x_He_III, r_T = ode_equation.compute_ode_residual(x_flux_vector,
                                                                                     x_state_vector,
                                                                                     x_time_vector,
                                                                                     parameter_vectors,
                                                                                     u_approximation)

            # [Issue] using log10(abs(prediction)) here introduces two problems:
            # 1. we lose sign information
            # 2. log doesn't have a minimum so we cant minimise it.
            # possible fixes: use sigmoid or tanh here, which helps us get rid of
            # both above issues
            out_x_H_II = torch.tanh(r_x_H_II)
            out_x_He_II = torch.tanh(r_x_He_II)
            out_x_He_III = torch.tanh(r_x_He_III)
            out_T = torch.tanh(r_T)

            loss_x_H_II = F.mse_loss(input=out_x_H_II, target=target_residuals, reduction='mean')
            loss_x_He_II = F.mse_loss(input=out_x_He_II, target=target_residuals, reduction='mean')
            loss_x_He_III = F.mse_loss(input=out_x_He_III, target=target_residuals, reduction='mean')
            loss_T = F.mse_loss(input=out_T, target=target_residuals, reduction='mean')

            loss_ode = loss_x_H_II + loss_x_He_II + loss_x_He_III + loss_T

            # compute the gradients
            loss_ode.backward()

            # update the parameters
            optimizer.step()
            # make the gradients zero
            optimizer.zero_grad()

            # compute sum of losses over all batches.
            epoch_loss += loss_ode.item()
            epoch_loss_x_H_II += loss_x_H_II.item()
            epoch_loss_T += loss_T.item()
            epoch_loss_x_He_II += loss_x_He_II.item()
            epoch_loss_x_He_III += loss_x_He_III.item()

        # compute average of all losses over all batches.
        epoch_loss /= len(train_loader)
        epoch_loss_x_H_II /= len(train_loader)
        epoch_loss_T /= len(train_loader)
        epoch_loss_x_He_II /= len(train_loader)
        epoch_loss_x_He_III /= len(train_loader)

        print("[Epoch %d/%d] [Train loss MSE: %e]" % (epoch, config.n_epochs, epoch_loss))

        train_loss_array = np.append(train_loss_array, epoch_loss)

        # log all losses to tensorboard
        data_log.log('loss_H_II', epoch_loss_x_H_II)
        data_log.log('loss_He_II', epoch_loss_x_He_II)
        data_log.log('loss_He_III', epoch_loss_x_He_III)
        data_log.log('loss_T', epoch_loss_T)
        data_log.log('Loss', epoch_loss)

        # update the tensorboard after every epoch
        data_log.update_data()

        # TODO: find a criteria to select the best model --- validation ---????
        # TODO: copy the best model based on validation....
        # TODO: implement validation here.

        # early stopping check
        if FORCE_STOP:
            print("\033[96m\033[1m\nStopping Early\033[0m\n")
            stopped_early = True
            epochs_trained = epoch
            break

    print("\033[96m\033[1m\nTraining complete\033[0m\n")

    data_log.close()

    # TODO: Save best model and loss functions
    # TODO: add final results to config and rewrite the actual file
    # TODO: plot loss functions
    # TODO: analysis


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='ML-RT - Cosmological radiative transfer with PINNs (PINN)')

    parser.add_argument('--out_dir', type=str, default='output', metavar='(string)',
                        help='Path to output directory, used for all plots and data products, default: ./output/')
    parser.add_argument('--pretraining_model_dir', type=str, default='../pretraining/output_pretraining/run_main', metavar='(string)',
                        help='Path of the run directory for the pre-trained model, default: ../pretraining/output_pretraining/run_main/')
    parser.add_argument('--run', type=str, default='', metavar='(string)',
                        help='Specific run name for the experiment, default: timestamp')

    # dataset and input data configuration
    parser.add_argument("--train_set_size", type=int, default=1024,
                        help="size of the randomly generated training set (default=1024)")
    parser.add_argument("--test_set_size", type=int, default=4096,
                        help="size of the randomly generated test set (default=4096)")
    parser.add_argument("--val_set_size", type=int, default=1024,
                        help="size of the randomly generated validation set (default=1024)")

    parser.add_argument("--batch_size", type=int, default=32,
                        help="size of the batches (default=32)")
    parser.add_argument("--len_latent_vector", type=int, default=8,
                        help="length of reduced SED vector")
    parser.add_argument("--len_state_vector", type=int, default=5,
                        help="length of state vector (Xi, T, t) to be concatenated with latent_vector")
    parser.add_argument("--len_SED_input", type=int, default=2000,
                        help="length of SED input for the model")

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
    parser.add_argument("--n_epochs", type=int, default=1000,
                        help="number of epochs of training")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="adam: learning rate, default=0.0001")
    parser.add_argument("--b1", type=float, default=0.9,
                        help="adam: beta1 - decay of first order momentum of gradient, default=0.9")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: beta2 - decay of first order momentum of gradient, default=0.999")

    # model configuration
    parser.add_argument("--model", type=str, default='MLP1', help="Model to use")

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
