import torch
from torch import nn


class AE1(nn.Module):
    def __init__(self, conf):
        super(AE1, self).__init__()

        def block(features_in, features_out, normalise=conf.batch_norm, dropout=conf.dropout):
            layers = [nn.Linear(features_in, features_out)]
            if normalise:
                layers.append(nn.BatchNorm1d(features_out))
            if dropout:
                layers.append(nn.Dropout(conf.dropout_value))

            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return layers

        self.encoder = nn.Sequential(
            *block(conf.len_SED_input, 512, normalise=False, dropout=False),
            *block(512, 128),
            *block(128, 64),
            nn.Linear(64, conf.len_latent_vector)
        )

        self.decoder = nn.Sequential(
            *block(conf.len_latent_vector, 64, normalise=False, dropout=False),
            *block(64, 128),
            *block(128, 512),
            nn.Linear(512, conf.len_SED_input)
        )

    def forward(self, x_flux_vector):
        """
        Inputs are:
        1) a vector containing the source flux, which is the SED multiplied with the optical depth, i.e.
           N(E) = exp(-tau(E))  * I(E)

        For a given flux vector
        (1) the encode function outputs a latent representation of (1).
        (2) and the decode function takes this latent representation and try to
        reconstruct (1) from it.

        Shape of the inputs:
        1)  (batch_size, len_flux_input)
        """
        latent_vector = self.encode(x_flux_vector)
        regenerated_flux_vector = self.decode(latent_vector)
        return regenerated_flux_vector

    def encode(self, flux_vector):
        return self.encoder(flux_vector)

    def decode(self, latent_rep):
        return self.decoder(latent_rep)
