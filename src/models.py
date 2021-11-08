import torch
from torch import nn


class MLP1(nn.Module):
    def __init__(self, conf):
        super(MLP_SAMPLE, self).__init__()

        def block(features_in, features_out, normalise=conf.batch_norm, dropout=conf.dropout):
            layers = [nn.Linear(features_in, features_out)]
            if normalise:
                layers.append(nn.BatchNorm1d(features_out))
            if dropout:
                layers.append(nn.Dropout(conf.dropout_value))

            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return layers

        self.NN_IE = nn.Sequential(
            *block(conf.len_SED_input, 512, normalise=False, dropout=False),
            *block(256, 128),
            *block(128, 64),
            *block(64, 16),
            nn.Linear(16, conf.len_latent_vector)
        )

        self.NN = nn.Sequential(
            *block(conf.len_state_vector + conf.len_latent_vector, 64, normalise=False, dropout=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            *block(1024, 512),
            *block(512, 128),
            *block(128, 64),
            nn.Linear(64, 4)
        )

    # Input:
    # x_SED: shape (batch_size, len_SED_input): vector representing spectral energy distribution of our source
    # x_state_vector: shape (batch_size, len_state_vector): vector representing state (Xi, tau, T, t)
    def forward(self, x_SED, x_state_vector):

        latent_vector = self.NN_IE(x_SED)
        # combine x_SED with latent_vector
        input_NN = torch.cat((x_state_vector, latent_vector), axis=1)
        out = self.NN(input_NN)

        return out
