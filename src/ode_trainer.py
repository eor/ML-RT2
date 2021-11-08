import argparse
import os

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from common.settings_ode import ode_parameter_limits as ps_ode
from common.settings_sed import p8_limits as ps_sed
from common.settings_sed import SED_ENERGY_MIN, SED_ENERGY_MAX, SED_ENERGY_DELTA
from common.utils import *
from sed import sed_numba
from models import *
from crt_equations import *

# check for CUDA
if torch.cuda.is_available():
    cuda = True
    device = torch.device("cuda")
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    cuda = False
    device = torch.device("cpu")
    torch.set_default_tensor_type(torch.FloatTensor)

def generate_training_data(config):
    train_set_size = config.train_set_size

    sed_vector = []
    haloMassLog = np.random.uniform(ps_sed[0][0], ps_sed[0][1], size=(train_set_size, 1))
    redshift = np.random.uniform(ps_sed[1][0], ps_sed[1][1], size=(train_set_size, 1))
    sourceAge = np.random.uniform(ps_sed[2][0], ps_sed[2][1], size=(train_set_size, 1))
    qsoAlpha = np.random.uniform(ps_sed[3][0], ps_sed[3][1], size=(train_set_size, 1))
    qsoEfficiency = np.random.uniform(ps_sed[4][0], ps_sed[4][1], size=(train_set_size, 1))
    starsEscFrac = np.random.uniform(ps_sed[5][0], ps_sed[5][1], size=(train_set_size, 1))
    starsIMFSlope = np.random.uniform(ps_sed[6][0], ps_sed[6][1], size=(train_set_size, 1))
    starsIMFMassMinLog = np.random.uniform(ps_sed[7][0], ps_sed[7][1], size=(train_set_size, 1))
    for i in range(train_set_size):
        # TODO: to verify correct variables used in the function -- Fabian??
        energies, inensities = sed_numba.generate_SED_IMF_PL(haloMass=haloMassLog[i][0],
                                redshift=redshift[i][0],
                                eLow=SED_ENERGY_MIN, eHigh=SED_ENERGY_MAX, N=2000,  logGrid=True,
                                starMassMin=5, starMassMax=500, imfBins=100, imfIndex=2.35, fEsc=starsEscFrac[i][0],
                                alpha=qsoAlpha[i][0], qsoEfficiency=qsoEfficiency[i][0],
                                targetSourceAge=sourceAge[i][0])
        sed_vector.append(energies)
    sed_vector = np.asarray(sed_vector)

    # sample SED vector
    x_H_II = np.random.uniform(ps_ode[0][0], ps_ode[0][1], size=(train_set_size, 1))
    x_He_II = np.random.uniform(ps_ode[1][0], ps_ode[1][1], size=(train_set_size, 1))
    x_He_III = np.random.uniform(ps_ode[2][0], ps_ode[2][1], size=(train_set_size, 1))
    T = np.random.uniform(ps_ode[3][0], ps_ode[3][1], size=(train_set_size, 1))
    tau = np.random.uniform(ps_ode[4][0], ps_ode[4][1], size=(train_set_size, 1))
    time = np.random.uniform(ps_ode[5][0], ps_ode[5][1], size=(train_set_size, 1))

    state_vector = np.concatenate((x_H_II, x_He_II, x_He_III, T, tau, time), axis=1)

    # sample target labels
    u_actual = np.zeros((train_set_size, 1))

    return sed_vector, state_vector, u_actual



# -----------------------------------------------------------------
#  Main
# -----------------------------------------------------------------
def main(config):
    # -----------------------------------------------------------------
    # create unique output path and run directories, save config
    # -----------------------------------------------------------------
    run_id = 'run_' + utils_get_current_timestamp()
    config.out_dir = os.path.join(config.out_dir, run_id)

    utils_create_output_dirs([config.out_dir])
    # utils_create_run_directories(config.out_dir, DATA_PRODUCTS_DIR, PLOT_DIR)
    utils_save_config_to_log(config)
    utils_save_config_to_file(config)

    # data_products_path = os.path.join(config.out_dir, DATA_PRODUCTS_DIR)
    # plot_path = os.path.join(config.out_dir, PLOT_DIR)

    # initialise the model
    u_approximation = MLP1(config)

    if cuda:
        model.cuda()

    # initialise the CRT equation
    ode_equation = CRT()

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

    print("\033[96m\033[1m\nTraining starts now\033[0m")
    for epoch in range(1, config.n_epochs + 1):
        # TODO: look for boundary conditions???

        x_SED, x_state_vector, target_residual = generate_training_data(config)

        # TODO: figure out: for what inputs do we need to set requires_grad=True
        x_SED = Variable(torch.from_numpy(x_SED).float(), requires_grad=False).to(device)
        x_state_vector = Variable(torch.from_numpy(x_state_vector).float(), requires_grad=False).to(device)
        target_residual = Variable(torch.from_numpy(target_residual).float(), requires_grad=False).to(device)

        # Loss based on CRT ODEs
        residual = ode_equation.compute_ode_residual(x_SED, x_state_vector, u_approximation)
        loss_ode = F.mse_loss(input=residual, target=target_residual, reduction='mean')

        # compute the gradients
        loss_ode.backward()
        # update the parameters
        optimizer.step()
        # make the gradients zero
        optimizer.zero_grad()

        print("[Epoch %d/%d] [Train loss MSE: %e]"
            % (epoch, config.n_epochs, loss_ode.item()))

        train_loss_array = np.append(train_loss_array, loss_ode.item())


        # TODO: find a criteria to select the best model --- validation ---????
        # TODO: copy the best model based on validation....

    print("\033[96m\033[1m\nTraining complete\033[0m\n")

    # TODO: Save best model and loss functions
    # TODO: add final results to config and rewrite the actual file
    # TODO: plot loss functions
    # TODO: analysis


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='ML-RT - Cosmological radiative transfer with PINNs (PINN)')

    parser.add_argument('--out_dir', type=str, default='output', metavar='(string)',
                    help='Path to output directory, used for all plots and data products, default: ./output/')

    parser.add_argument("--len_SED_input", type=int, default=2000,
                        help="length of SED input for the model")
    parser.add_argument("--len_latent_vector", type=int, default=8,
                        help="length of reduced SED vector")
    parser.add_argument("--len_state_vector", type=int, default=6,
                        help="length of state vector (Xi, T, tau, t) to be concatenated with latent_vector")
    parser.add_argument("--train_set_size", type=int, default=128,
                        help="size of the randomly generated training set (default=128)")

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

    # print summary
    print("\nUsed parameters:\n")
    for arg in vars(my_config):
        print("\t", arg, getattr(my_config, arg))



    # run main program
    main(my_config)
