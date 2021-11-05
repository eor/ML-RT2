import torch
import numpy as np
import deepxde as dde


class DIFFUSION_1D:

    def __init__(self):
        pass

    def pde(self, x, y):
        dy_t = dde.grad.jacobian(y, x, j=1)
        dy_xx = dde.grad.hessian(y, x, j=0)

        # Backend pytorch
        return (
            dy_t
            - dy_xx
            + torch.exp(-x[:, 1:])
            * (torch.sin(np.pi * x[:, 0:1]) - np.pi ** 2 * torch.sin(np.pi * x[:, 0:1]))
        )

    def func(self, x):
        return np.sin(np.pi * x[:, 0:1]) * np.exp(-x[:, 1:])
