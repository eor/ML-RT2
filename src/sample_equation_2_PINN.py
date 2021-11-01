import argparse
import os

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from models.mlp import *
from equations.sample_equation_2 import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# check for CUDA
if torch.cuda.is_available():
    cuda = True
    device = torch.device("cuda")
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    cuda = False
    device = torch.device("cpu")
    torch.set_default_tensor_type(torch.FloatTensor)

# # -----------------------------------------------------------------
# #  Test Metric
# # -----------------------------------------------------------------
# if cuda:
#     soft_dtw_loss = SoftDTW_CUDA(use_cuda=True, gamma=0.1)
# else:
#     soft_dtw_loss = SoftDTW_CPU(use_cuda=False, gamma=0.1)
#
#
# def pinn_test_metric(func, gen_x, real_x, config):
#     if func == 'DTW':
#         # profile tensors are of shape [batch size, profile length]
#         # soft dtw wants input of shape [batch size, 1, profile length]
#         if len(gen_x.size()) != 3:
#             loss = soft_dtw_loss(gen_x.unsqueeze(1), real_x.unsqueeze(1)).mean()
#         else:
#             loss = soft_dtw_loss(gen_x, real_x).mean()
#     else:
#         loss = F.mse_loss(input=gen_x, target=real_x, reduction='mean')
#     return loss


def generate_data_boundary_conditions(config):
    train_set_size = config.train_set_size
    # Data from Boundary Conditions
    # u(x,0) = 6e^(-3x)
    # BC just gives us data points for training
    # BC tells us that for any x in range[0,2] and time=0, the value of u is given by 6e^(-3x)
    # Take say train_set_size random numbers of x
    x_bc = np.random.uniform(low=0.0, high=2.0, size=(train_set_size, 1))
    t_bc = np.zeros((train_set_size, 1))
    # compute u based on BC
    u_bc = 6*np.exp(-3*x_bc)

    return x_bc, t_bc, u_bc


def generate_data_pde(config):
    train_set_size = config.train_set_size
    x_actual = np.random.uniform(low=0.0, high=2.0, size=(train_set_size, 1))
    t_actual = np.random.uniform(low=0.0, high=1.0, size=(train_set_size, 1))
    u_actual = np.zeros((train_set_size, 1))

    return x_actual, t_actual, u_actual


def evaluate(model):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    x = np.arange(0, 2, 0.02)
    t = np.arange(0, 1, 0.02)
    ms_x, ms_t = np.meshgrid(x, t)

    # Because meshgrid is used, we need to do the following adjustment
    x = np.ravel(ms_x).reshape(-1, 1)
    t = np.ravel(ms_t).reshape(-1, 1)

    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    pt_t = Variable(torch.from_numpy(t).float(), requires_grad=True).to(device)
    pt_u = model(pt_x, pt_t)
    u = pt_u.data.cpu().numpy()
    ms_u = u.reshape(ms_x.shape)

    surf = ax.plot_surface(ms_x, ms_t, ms_u, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def main(config):

    # initialise the equation to solve
    equation = Equation2()

    # we assume MLP as a solution to our problem
    u_approximation = MLP_SAMPLE(config)
    u_approximation.to(device)

    # set up optimizers
    optimizer = torch.optim.Adam(
        u_approximation.parameters(),
        lr=config.lr,
        betas=(config.b1, config.b2)
    )

    print("\033[96m\033[1m\nTraining starts now\033[0m")
    for epoch in range(1, config.n_epochs + 1):
        x_bc, t_bc, u_bc = generate_data_boundary_conditions(config)
        x_bc = Variable(torch.from_numpy(x_bc).float(), requires_grad=False).to(device)
        t_bc = Variable(torch.from_numpy(t_bc).float(), requires_grad=False).to(device)
        u_bc = Variable(torch.from_numpy(u_bc).float(), requires_grad=False).to(device)

        # Loss based on boundary conditions
        u_prediction_bc = u_approximation(x_bc, t_bc)
        loss_bc = F.mse_loss(input=u_bc, target=u_prediction_bc, reduction='mean')

        x, t, u = generate_data_pde(config)
        x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
        t = Variable(torch.from_numpy(t).float(), requires_grad=True).to(device)
        u = Variable(torch.from_numpy(u).float(), requires_grad=True).to(device)

        # Loss based on PDE
        u_prediction = equation.pde(x, t, u_approximation)
        loss_pde = F.mse_loss(input=u, target=u_prediction, reduction='mean')

        loss = loss_pde + loss_bc
        # compute the gradients
        loss.backward()
        # update the parameters
        optimizer.step()
        # make the gradients zero
        optimizer.zero_grad()

        if epoch % 1000 == 0:
            with torch.autograd.no_grad():
                print(epoch, "Training Loss:", loss.data)

    torch.save(u_approximation.state_dict(), "model_uxt.pt")
    evaluate(u_approximation)


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='ML-RT - Cosmological radiative transfer with PINNs (PINN)')

    parser.add_argument("--start_profile", type=int, default=0,
                        help="initial value for r")
    parser.add_argument("--end_profile", type=int, default=1500,
                        help="final point for r")
    parser.add_argument("--n_parameters", type=int, default=8,
                        help="number of RT parameters (5 or 8)")

    # network optimisation
    parser.add_argument("--n_epochs", type=int, default=10000,
                        help="number of epochs of training")
    # still relevant---?
    parser.add_argument("--batch_size", type=int, default=32,
                        help="size of the batches (default=32)")
    parser.add_argument("--train_set_size", type=int, default=2048,
                        help="size of the randomly generated training set (default=2048)")

    parser.add_argument("--lr", type=float, default=0.0001,
                        help="adam: learning rate, default=0.0001")
    parser.add_argument("--b1", type=float, default=0.9,
                        help="adam: beta1 - decay of first order momentum of gradient, default=0.9")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: beta2 - decay of first order momentum of gradient, default=0.999")

    # analysis
    parser.add_argument("--analysis", dest='analysis', action='store_true',
                        help="automatically generate some plots (default)")
    parser.add_argument("--no-analysis", dest='analysis', action='store_false', help="do not run analysis")
    parser.set_defaults(analysis=True)

    # analysis
    parser.add_argument("--evaluation", dest='evaluate', action='store_true',
                        help="test on the given dataset")
    parser.add_argument("--no-evaluation", dest='evaluate', action='store_false', help="do not evaluate model")
    # by default, evaluate the model on test set
    parser.set_defaults(evaluate=True)

    my_config = parser.parse_args()

    # sanity checks
    # if my_config.data_dir is None:
    #     print('\nError: Parameter data_dir must not be empty. Exiting.\n')
    #     argparse.ArgumentParser().print_help()
    #     exit(1)
    #
#     if my_config.n_parameters not in [5, 8]:
#         print(
# '\nError: Number of parameters can currently only be either 5 or 8. Exiting.\n')
#         argparse.ArgumentParser().print_help()
#         exit(1)
#
#     if my_config.n_parameters == 5:
#         parameter_limits = ps.p5_limits
#         parameter_names_latex = ps.p5_names_latex
#
#     if my_config.n_parameters == 8:
#         parameter_limits = ps.p8_limits
#         parameter_names_latex = ps.p8_names_latex
#
    my_config.model = 'MLP1'

    # print summary
    print("\nUsed parameters:\n")
    for arg in vars(my_config):
        print("\t", arg, getattr(my_config, arg))

    # run main program
    main(my_config)
