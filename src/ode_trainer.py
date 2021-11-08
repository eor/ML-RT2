import argparse
import os

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import common.settings_ode as ps_ode
import common.settings_sed as ps_sed



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

    # TODO: sample state vector
    state_vector = None
    # TODO: sample SED vector
    sed_vector = None
    # TODO: generate target labels
    u_actual = None
    return state_vector, sed_vector, u_actual



# -----------------------------------------------------------------
#  Main
# -----------------------------------------------------------------
def main(config):
    # -----------------------------------------------------------------
    # create unique output path and run directories, save config
    # -----------------------------------------------------------------
    run_id = 'run_' + utils_get_current_timestamp()
    config.out_dir = os.path.join(config.out_dir, run_id)

    # utils_create_run_directories(config.out_dir, DATA_PRODUCTS_DIR, PLOT_DIR)
    utils_save_config_to_log(config)
    utils_save_config_to_file(config)

    # data_products_path = os.path.join(config.out_dir, DATA_PRODUCTS_DIR)
    # plot_path = os.path.join(config.out_dir, PLOT_DIR)

    u_approximation = MLP1(config)

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
    train_loss_array = np.empty(0)
    val_loss_mse_array = np.empty(0)
    val_loss_dtw_array = np.empty(0)

    print("\033[96m\033[1m\nTraining starts now\033[0m")
    for epoch in range(1, config.n_epochs + 1):
        # TODO: look for boundary conditions???

        x_SED, x_state_vector, u_actual = generate_training_data(config)

        # TODO: figure out: for what inputs do we need to set requires_grad=True
        x_SED = Variable(torch.from_numpy(x_SED).float(), requires_grad=False).to(device)
        x_state_vector = Variable(torch.from_numpy(x_state_vector).float(), requires_grad=False).to(device)
        u_actual = Variable(torch.from_numpy(u_actual).float(), requires_grad=False).to(device)

        # Loss based on CRT ODEs
        u_prediction = u_approximation(x_SED, x_state_vector)
        loss_ode = F.mse_loss(input=u_actual, target=u_prediction, reduction='mean')

        # compute the gradients
        loss_ode.backward()
        # update the parameters
        optimizer.step()
        # make the gradients zero
        optimizer.zero_grad()
        train_loss_array = np.append(train_loss_array, loss_ode.item())

        print("[Epoch %d/%d] [Train loss MSE: %e]"
                    % (epoch, config.n_epochs, train_loss))

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

    parser.add_argument("--len_SED_input", type=int, default=1024,
                        help="length of SED input for the model")
    parser.add_argument("--len_latent_vector", type=int, default=8,
                        help="length of reduced SED vector")
    parser.add_argument("--len_state_vector", type=int, default=6,
                        help="length of state vector (Xi, tau, T, t) to be concatenated with latent_vector")
    parser.add_argument("--train_set_size", type=int, default=2048,
                        help="size of the randomly generated training set (default=2048)")

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
