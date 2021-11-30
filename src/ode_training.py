import argparse
import os
import signal

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from common.settings_ode import ode_parameter_limits as ps_ode
from common.settings_sed import p8_limits as ps_sed
from common.settings_sed import SED_ENERGY_MIN, SED_ENERGY_MAX, SED_ENERGY_DELTA
from common.utils import *
from common.physics import *
from common.settings_crt import *
from common.settings import *
from common.data_log import *
from sed import sed_numba
from models import *
from ode import *
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


def generate_tau_training(energies_vector):
    """
    The generated training data produces two inputs to the neural network: a state vector and the source flux, which
    represents the SED that is attenuated by the IGM in between the source and a given point of interest.
    The attenuation can be expressed a factor exp(-tau(E,r,t)), where tau is the optical depth.

    This function is meant to serve as a realistic approximation to tau(E). Time dependence can be discounted and
    the radial dependence can be interpreted as a column density, since for a discreet computing grid tau(E)
    can be written as

    sum_i [ sigma_i(E) sum_r {n_i(r) delta_r } ] = sum_i [sigma_i(E)] N_i,

    where i = (H I, He I, He II) and N_i = sum_r {n_i(r) delta_r} is called the column density

    We pick suitable column densities, compute tau and return a vector tau(energies)
    """

    train_set_size = energies_vector.shape[0]
    len_energy_vector = energies_vector.shape[1]

    sigmas_H_I = np.zeros(len_energy_vector)
    sigmas_He_I = np.zeros(len_energy_vector)
    sigmas_He_II = np.zeros(len_energy_vector)
    tau_training_vector = np.zeros((train_set_size, len_energy_vector))

    # obtain sigmas for the energy_vector
    # we have same energy vector for every sample in train_set
    # Hence, can be computed just once
    physics = Physics.getInstance()
    physics.set_energy_vector(energies_vector[0])
    sigmas_H_I = physics.get_photo_ionisation_cross_section_hydrogen()
    sigmas_He_I = physics.get_photo_ionisation_cross_section_helium1()
    sigmas_He_II = physics.get_photo_ionisation_cross_section_helium2()

    # define column densities
    limit_lower = -20.0
    limit_upper = 0.0

    # generate tau for our training
    for t in range(train_set_size):
        N_H_I = 10**(random() * (limit_upper - limit_lower) + limit_lower)
        N_He_I = 10**(random() * (limit_upper - limit_lower) + limit_lower)
        N_He_II = 10**(random() * (limit_upper - limit_lower) + limit_lower)

        tau_training_vector[t] = sigmas_H_I * N_H_I + sigmas_He_I * N_He_I + sigmas_He_II * N_He_II

    return tau_training_vector


def generate_training_data(config):
    """
    Explanation here please.
    """
    # retrieve train_set_size
    train_set_size = config.train_set_size

    # sample SED vector
    intensities_vector = []
    energies_vector = []
    haloMassLog = np.random.uniform(ps_sed[0][0], ps_sed[0][1], size=(train_set_size, 1))
    redshift = np.random.uniform(ps_sed[1][0], ps_sed[1][1], size=(train_set_size, 1))
    sourceAge = np.random.uniform(ps_sed[2][0], ps_sed[2][1], size=(train_set_size, 1))
    qsoAlpha = np.random.uniform(ps_sed[3][0], ps_sed[3][1], size=(train_set_size, 1))
    qsoEfficiency = np.random.uniform(ps_sed[4][0], ps_sed[4][1], size=(train_set_size, 1))
    starsEscFrac = np.random.uniform(ps_sed[5][0], ps_sed[5][1], size=(train_set_size, 1))
    starsIMFSlope = np.random.uniform(ps_sed[6][0], ps_sed[6][1], size=(train_set_size, 1))
    starsIMFMassMin = np.random.uniform(ps_sed[7][0], ps_sed[7][1], size=(train_set_size, 1))

    parameter_vector = np.concatenate((haloMassLog, redshift, sourceAge, qsoAlpha,
     qsoEfficiency, starsEscFrac, starsIMFSlope, starsIMFMassMin), axis=1)

    for i in range(train_set_size):
        energies, intensities = sed_numba.generate_SED_IMF_PL(halo_mass=haloMassLog[i][0],
                                                              redshift=redshift[i][0],
                                                              eLow=SED_ENERGY_MIN,
                                                              eHigh=SED_ENERGY_MAX,
                                                              N=2000,  logGrid=True,
                                                              starMassMin=starsIMFMassMin[i][0],
                                                              starMassMax=500,
                                                              imfBins=50,
                                                              imfIndex=starsIMFSlope[i][0],
                                                              fEsc=starsEscFrac[i][0],
                                                              alpha=qsoAlpha[i][0],
                                                              qsoEfficiency=qsoEfficiency[i][0],
                                                              targetSourceAge=sourceAge[i][0])
        intensities_vector.append(intensities)
        energies_vector.append(energies)

    # convert lists to numpy arrays
    intensities_vector = np.asarray(intensities_vector)
    energies_vector = np.asarray(energies_vector)
    # obtain tau from energies_vector
    tau = generate_tau_training(energies_vector)
    # obtain flux_vector from intensities_vector by multiplying with tau
    flux_vector = np.multiply(intensities_vector, tau)

    # sample state vector
    x_H_II = np.random.uniform(ps_ode[0][0], ps_ode[0][1], size=(train_set_size, 1))
    x_He_II = np.random.uniform(ps_ode[1][0], ps_ode[1][1], size=(train_set_size, 1))
    x_He_III = np.random.uniform(ps_ode[2][0], ps_ode[2][1], size=(train_set_size, 1))
    T = np.random.uniform(ps_ode[3][0], ps_ode[3][1], size=(train_set_size, 1))

    # [TODO]: verify this. upper range will targetSourceAge or sourceAge upper bound from settings?
    lower_bound_time = ps_ode[5][0] * np.ones(shape=(train_set_size, 1))
    upper_bound_time = sourceAge.copy()
    time_vector = np.random.uniform(lower_bound_time, upper_bound_time, size=(train_set_size, 1))

    state_vector = np.concatenate((x_H_II, x_He_II, x_He_III, T), axis=1)

    # sample target labels
    precision_upto_digits = 4
    u_actual = -1 * precision_upto_digits * np.ones((train_set_size))

    return flux_vector, state_vector, time_vector, u_actual, parameter_vector, energies_vector

def force_stop_signal_handler(sig, frame):
    global FORCE_STOP
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

        # TODO: look for boundary conditions???
        x_flux_vector, x_state_vector, x_time_vector, target_residual, parameter_vector, energies_vector = generate_training_data(config)

        # update the data in this physics module
        physics.set_energy_vector(energies_vector[0])
        physics.set_flux_vector(x_flux_vector)

        # TODO: figure out: for what inputs do we need to set requires_grad=True
        x_flux_vector = Variable(torch.from_numpy(x_flux_vector).float(), requires_grad=True).to(device)
        x_state_vector = Variable(torch.from_numpy(x_state_vector).float(), requires_grad=True).to(device)
        x_time_vector = Variable(torch.from_numpy(x_time_vector).float(), requires_grad=True).to(device)
        target_residual = Variable(torch.from_numpy(target_residual).float(), requires_grad=True).to(device)
        parameter_vector = Variable(torch.from_numpy(parameter_vector).float(), requires_grad=True).to(device)

        # Loss based on CRT ODEs
        residual = ode_equation.compute_ode_residual(x_flux_vector, x_state_vector, x_time_vector, parameter_vector, u_approximation)
        log_residual = torch.log10(torch.abs(residual))
        # loss: [log(residual) - precision]^2
        loss_ode = F.mse_loss(input=log_residual, target=target_residual, reduction='mean')

        # compute the gradients
        loss_ode.backward()

        # update the parameters
        optimizer.step()
        # make the gradients zero
        optimizer.zero_grad()

        print("[Epoch %d/%d] [Train loss MSE: %e] [residual: %e]"
            % (epoch, config.n_epochs, loss_ode.item(), residual.mean().item()))

        train_loss_array = np.append(train_loss_array, loss_ode.item())

        # log data to the data log
        data_log.log('Residual', residual.mean().item())
        data_log.log('Loss', loss_ode.item())

        # update the tensorboard after every epoch
        data_log.update_data()

        # TODO: find a criteria to select the best model --- validation ---????
        # TODO: copy the best model based on validation....

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
    parser.add_argument('--run', type=str, default='', metavar='(string)',
                        help='Specific run name for the experiment, default: ./output/timestamp')

    parser.add_argument("--len_SED_input", type=int, default=2000,
                        help="length of SED input for the model")
    parser.add_argument("--len_latent_vector", type=int, default=8,
                        help="length of reduced SED vector")
    parser.add_argument("--len_state_vector", type=int, default=5,
                        help="length of state vector (Xi, T, t) to be concatenated with latent_vector")
    parser.add_argument("--train_set_size", type=int, default=128,
                        help="size of the randomly generated training set (default=128)")

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
    parser.add_argument("--n_epochs", type=int, default=10000,
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
