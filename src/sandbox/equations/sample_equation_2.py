import torch
import numpy as np
import deepxde as dde


class Equation2:
    """
    Equation2:
        f = du/dx - 2du/dt - u
    """

    def __init__(self):
        pass

    def pde(self, x, t, u_approximation):

        # the dependent variable u is given by the network based on independent variables x,t
        u = u_approximation(x, t)

        # Based on our function f = du/dx - 2du/dt - u,
        # we need du/dx and du/dt
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        pde = u_x - 2*u_t - u

        return pde
