import torch
from torch import nn

class MLP_SAMPLE(nn.Module):
    def __init__(self, conf):
        super(MLP_SAMPLE, self).__init__()

        def block(features_in, features_out, normalise=False, dropout=False):

            layers = [nn.Linear(features_in, features_out)]

            # Different order of BN, Dropout, and non-linearity should be explored!
            # From the literature it seems like there is no canonical way of doing it.
            if normalise:
                layers.append(nn.BatchNorm1d(features_out))

            if dropout:
                layers.append(nn.Dropout(conf.dropout_value))

            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return layers

        self.model = nn.Sequential(
            *block(2, 5, normalise=False, dropout=False),
            *block(5, 5),
            *block(5, 5),
            *block(5, 5),
            *block(5, 5),
            nn.Linear(5,1)
        )

    def forward(self, x, t):
        # combined two arrays of 1 columns each to one array of 2 columns
        inputs = torch.cat([x,t],axis=1)
        return self.model(inputs)
