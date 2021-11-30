import torch
from torch import nn


class MLP1(nn.Module):
    def __init__(self, conf):
        super(MLP1, self).__init__()

        def block(features_in, features_out, normalise=conf.batch_norm, dropout=conf.dropout):
            layers = [nn.Linear(features_in, features_out)]
            if normalise:
                layers.append(nn.BatchNorm1d(features_out))
            if dropout:
                layers.append(nn.Dropout(conf.dropout_value))

            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return layers

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.uniform_(m.weight, a=0.0, b=1.0)
                m.bias.data.fill_(0.01)

        self.NN_flux = nn.Sequential(
            *block(conf.len_SED_input, 512, normalise=False, dropout=False),
            *block(512, 128),
            *block(128, 64),
            *block(64, 16),
            nn.Linear(16, conf.len_latent_vector)
        )

        self.NN = nn.Sequential(
            *block(conf.len_state_vector + conf.len_latent_vector, 64, normalise=False, dropout=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 128),
            *block(128, 64),
            nn.Linear(64, 4)
        )

        # self.NN_flux.apply(init_weights)
        # self.NN.apply(init_weights)



    def forward(self, x_flux_vector, x_state_vector, time_vector):
        """
        Inputs are:
        1) a vector containing the source flux, which is the SED multiplied with the optical depth, i.e.
           N(E) = exp(-tau(E))  * I(E)

        2) a state vector, which contains ionisation fractions, temperature and time

        For a given flux vector (1) the NN_flux network outputs a latent representation of (1) which is combined
        with the state vector (2) to form the input to the second network, i.e. NN.

        Shape of the inputs:
        1)  (batch_size, len_flux_input)

        2) (batch_size, len_state_vector):
        """

        latent_vector = self.NN_flux(x_flux_vector)
        concat_input = torch.cat((x_state_vector, time_vector, latent_vector), axis=1)
        output = self.NN(concat_input)

        # [Issue] high order of values almost lead to constant values near the extremes of sigmoid
        # possible fixes: use batch norm by default or somehow normalise the input to be small maybe
        x_H_II_prediction = torch.sigmoid(output[:, 0])
        x_He_II_prediction = torch.sigmoid(output[:, 1])
        x_He_III_prediction = torch.sigmoid(output[:, 2])
        T_prediction = torch.pow(10, 13 * torch.sigmoid(output[:, 3]))

        return x_H_II_prediction, x_He_II_prediction, x_He_III_prediction, T_prediction

# separate layers
