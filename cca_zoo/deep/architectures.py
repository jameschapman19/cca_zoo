from math import sqrt
from typing import Iterable

import torch
from torch import nn


class BaseEncoder(nn.Module):
    def __init__(self, latent_dimensions: int, variational: bool = False):
        super(BaseEncoder, self).__init__()
        self.latent_dimensions = latent_dimensions
        self.variational = variational

    def create_variational_layers(self, in_features):
        self.fc_mu = torch.nn.Linear(in_features, self.latent_dimensions)
        self.fc_var = torch.nn.Linear(in_features, self.latent_dimensions)

    def forward_variational(self, x):
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        return mu, logvar


class Encoder(BaseEncoder):
    def __init__(
        self,
        latent_dimensions: int,
        feature_size: int,
        variational: bool = False,
        layer_sizes: tuple = None,
        activation=nn.LeakyReLU(),
        dropout=0,
    ):
        super(Encoder, self).__init__(latent_dimensions, variational=variational)
        if layer_sizes is None:
            layer_sizes = (128,)
        layer_sizes = (feature_size,) + layer_sizes + (latent_dimensions,)
        layers = [
            nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(layer_sizes[i], layer_sizes[i + 1]),
                activation,
            )
            for i in range(len(layer_sizes) - 2)
        ]
        self.layers = nn.Sequential(*layers)

        if self.variational:
            self.create_variational_layers(layer_sizes[-2])
        else:
            self.fc = nn.Linear(layer_sizes[-2], layer_sizes[-1])

    def forward(self, x):
        x = self.layers(x)
        if self.variational:
            return self.forward_variational(x)
        return self.fc(x)


class LinearEncoder(BaseEncoder):
    def __init__(
        self, latent_dimensions: int, feature_size: int, variational: bool = False
    ):
        super(LinearEncoder, self).__init__(latent_dimensions, variational=variational)
        if self.variational:
            self.create_variational_layers(feature_size)
        else:
            self.fc = torch.nn.Linear(feature_size, latent_dimensions)

    def forward(self, x):
        if self.variational:
            return self.forward_variational(x)
        return self.fc(x)


class CNNEncoder(BaseEncoder):
    def __init__(
        self,
        latent_dimensions: int,
        feature_size: Iterable,
        variational: bool = False,
        channels: tuple = None,
        kernel_sizes: tuple = None,
        strides: tuple = None,
        paddings: tuple = None,
        activation=nn.LeakyReLU(),
        dropout=0,
    ):
        super(CNNEncoder, self).__init__(latent_dimensions, variational=variational)

        default_len = 2 if channels is None else len(channels)
        if channels is None:
            channels = (1, 1)
        kernel_sizes = kernel_sizes or (5,) * default_len
        strides = strides or (1,) * default_len
        paddings = paddings or (2,) * default_len

        self.conv_layers = self._build_conv_layers(
            channels, kernel_sizes, strides, paddings, activation
        )

        final_channels = channels[-1]
        final_size = feature_size[0]  # Assuming square input
        linear_input_size = final_channels * final_size * final_size

        if self.variational:
            self.create_variational_layers(linear_input_size)
        else:
            self.fc = nn.Sequential(
                nn.Dropout(p=dropout), nn.Linear(linear_input_size, latent_dimensions)
            )

    def _build_conv_layers(self, channels, kernel_sizes, strides, paddings, activation):
        layers = []
        current_channels = 1
        for idx in range(len(channels)):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=current_channels,
                        out_channels=channels[idx],
                        kernel_size=kernel_sizes[idx],
                        stride=strides[idx],
                        padding=paddings[idx],
                    ),
                    activation,
                )
            )
            current_channels = channels[idx]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        if self.variational:
            return self.forward_variational(x)
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dimensions: int,
        feature_size: int = 784,
        layer_sizes: tuple = None,
        activation=nn.LeakyReLU(),
        dropout=0,
    ):
        super(Decoder, self).__init__()
        if layer_sizes is None:
            layer_sizes = (128,)
        layer_sizes = (latent_dimensions,) + layer_sizes + (feature_size,)
        layers = [
            nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(layer_sizes[i], layer_sizes[i + 1]),
                activation,
            )
            for i in range(len(layer_sizes) - 1)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class CNNDecoder(nn.Module):
    def __init__(
        self,
        latent_dimensions: int,
        feature_size: Iterable = (28, 28),
        channels: tuple = None,
        kernel_sizes: tuple = None,
        strides: tuple = None,
        paddings: tuple = None,
        activation=nn.LeakyReLU(),
        dropout=0,
    ):
        super(CNNDecoder, self).__init__()

        default_len = 2 if channels is None else len(channels)
        if channels is None:
            channels = (1, 1)
        kernel_sizes = kernel_sizes or (5,) * default_len
        strides = strides or (1,) * default_len
        paddings = paddings or (2,) * default_len

        self.conv_layers = self._build_transpose_conv_layers(
            feature_size, channels, kernel_sizes, strides, paddings, activation
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(
                latent_dimensions, channels[0] * feature_size[0] * feature_size[1]
            ),
            activation,
        )

    def _build_transpose_conv_layers(
        self, feature_size, channels, kernel_sizes, strides, paddings, activation
    ):
        layers = []
        for idx in reversed(range(len(channels) - 1)):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=channels[idx],
                        out_channels=channels[idx + 1],
                        kernel_size=kernel_sizes[idx],
                        stride=strides[idx],
                        padding=paddings[idx],
                    ),
                    activation,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc_layer(x)
        x = x.view(
            x.size(0),
            self.conv_layers[0][0].in_channels,
            int(sqrt(x.size(1))),
            int(sqrt(x.size(1))),
        )
        return self.conv_layers(x)


class LinearDecoder(nn.Module):
    def __init__(self, latent_dimensions: int, feature_size: int):
        super(LinearDecoder, self).__init__(latent_dimensions)
        self.linear = torch.nn.Linear(latent_dimensions, feature_size)

    def forward(self, x):
        out = self.linear(x)
        return out
