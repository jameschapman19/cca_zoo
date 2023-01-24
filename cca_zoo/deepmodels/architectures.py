from abc import abstractmethod
from math import sqrt
from typing import Iterable

import torch
from torch import nn


class _BaseEncoder(torch.nn.Module):
    @abstractmethod
    def __init__(self, latent_dims: int, variational: bool = False):
        super(_BaseEncoder, self).__init__()
        self.variational = variational
        self.latent_dims = latent_dims

    @abstractmethod
    def forward(self, x):
        pass


class _BaseDecoder(torch.nn.Module):
    @abstractmethod
    def __init__(self, latent_dims: int):
        super(_BaseDecoder, self).__init__()
        self.latent_dims = latent_dims

    @abstractmethod
    def forward(self, x):
        pass


class Encoder(_BaseEncoder):
    def __init__(
        self,
        latent_dims: int,
        variational: bool = False,
        feature_size: int = 784,
        layer_sizes: tuple = None,
        activation=nn.LeakyReLU(),
        dropout=0,
    ):
        super(Encoder, self).__init__(latent_dims, variational=variational)
        if layer_sizes is None:
            layer_sizes = (128,)
        layer_sizes = (feature_size,) + layer_sizes + (latent_dims,)
        layers = []
        # other layers
        for l_id in range(len(layer_sizes) - 2):
            layers.append(
                torch.nn.Sequential(
                    nn.Dropout(p=dropout),
                    torch.nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                    activation,
                )
            )
        self.layers = torch.nn.Sequential(*layers)

        if self.variational:
            self.fc_mu = torch.nn.Sequential(
                nn.Dropout(p=dropout), torch.nn.Linear(layer_sizes[-2], layer_sizes[-1])
            )
            self.fc_var = torch.nn.Sequential(
                nn.Dropout(p=dropout), torch.nn.Linear(layer_sizes[-2], layer_sizes[-1])
            )
        else:
            self.fc = torch.nn.Sequential(
                nn.Dropout(p=dropout), torch.nn.Linear(layer_sizes[-2], layer_sizes[-1])
            )

    def forward(self, x):
        x = self.layers(x)
        if self.variational:
            mu = self.fc_mu(x)
            logvar = self.fc_var(x)
            return mu, logvar
        else:
            x = self.fc(x)
            return x


class Decoder(_BaseDecoder):
    def __init__(
        self,
        latent_dims: int,
        feature_size: int = 784,
        layer_sizes: tuple = None,
        activation=nn.LeakyReLU(),
        dropout=0,
    ):
        super(Decoder, self).__init__(latent_dims)
        if layer_sizes is None:
            layer_sizes = (128,)
        layer_sizes = (latent_dims,) + layer_sizes + (feature_size,)
        layers = []
        for l_id in range(len(layer_sizes) - 1):
            layers.append(
                torch.nn.Sequential(
                    nn.Dropout(p=dropout),
                    torch.nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                    activation,
                )
            )
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class CNNEncoder(_BaseEncoder):
    def __init__(
        self,
        latent_dims: int,
        variational: bool = False,
        feature_size: Iterable = (28, 28),
        channels: tuple = None,
        kernel_sizes: tuple = None,
        stride: tuple = None,
        padding: tuple = None,
        activation=nn.LeakyReLU(),
        dropout=0,
    ):
        super(CNNEncoder, self).__init__(latent_dims, variational=variational)
        if channels is None:
            channels = (1, 1)
        if kernel_sizes is None:
            kernel_sizes = (5,) * (len(channels))
        if stride is None:
            stride = (1,) * (len(channels))
        if padding is None:
            padding = (2,) * (len(channels))
        # assume square input
        conv_layers = []
        current_size = feature_size[0]
        current_channels = 1
        for l_id in range(len(channels) - 1):
            conv_layers.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=current_channels,  # input height
                        out_channels=channels[l_id],  # n_filters
                        kernel_size=kernel_sizes[l_id],  # filter size
                        stride=stride[l_id],  # filter movement/step
                        padding=padding[l_id],
                        # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if
                        # stride=1
                    ),  # output shape (out_channels, current_size, current_size)
                    activation,  # activation
                )
            )
            current_size = current_size
            current_channels = channels[l_id]

        if self.variational:
            self.fc_mu = torch.nn.Sequential(
                nn.Dropout(p=dropout),
                torch.nn.Linear(
                    int(current_size * current_size * current_channels), latent_dims
                ),
            )
            self.fc_var = torch.nn.Sequential(
                nn.Dropout(p=dropout),
                torch.nn.Linear(
                    int(current_size * current_size * current_channels), latent_dims
                ),
            )
        else:
            self.fc = torch.nn.Sequential(
                nn.Dropout(p=dropout),
                torch.nn.Linear(
                    int(current_size * current_size * current_channels), latent_dims
                ),
            )
        self.conv_layers = torch.nn.Sequential(*conv_layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape((x.shape[0], -1))
        if self.variational:
            mu = self.fc_mu(x)
            logvar = self.fc_var(x)
            return mu, logvar
        else:
            x = self.fc(x)
            return x


class CNNDecoder(_BaseDecoder):
    def __init__(
        self,
        latent_dims: int,
        feature_size: Iterable = (28, 28),
        channels: tuple = None,
        kernel_sizes=None,
        strides=None,
        paddings=None,
        activation=nn.LeakyReLU(),
        dropout=0,
    ):
        super(CNNDecoder, self).__init__(latent_dims)
        if channels is None:
            channels = (1, 1)
        if kernel_sizes is None:
            kernel_sizes = (5,) * len(channels)
        if strides is None:
            strides = (1,) * len(channels)
        if paddings is None:
            paddings = (2,) * len(channels)
        conv_layers = []
        current_channels = 1
        current_size = feature_size[0]
        for l_id, (channel, kernel, stride, padding) in reversed(
            list(enumerate(zip(channels, kernel_sizes, strides, paddings)))
        ):
            conv_layers.append(
                torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(
                        in_channels=channel,  # input height
                        out_channels=current_channels,
                        kernel_size=kernel_sizes[l_id],
                        stride=strides[l_id],  # filter movement/step
                        padding=paddings[l_id],
                        # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if
                        # stride=1
                    ),
                    activation,
                )
            )
            current_size = current_size
            current_channels = channel

        # reverse layers as constructed in reverse
        self.conv_layers = torch.nn.Sequential(*conv_layers[::-1])
        self.fc_layer = torch.nn.Sequential(
            nn.Dropout(p=dropout),
            torch.nn.Linear(
                latent_dims, int(current_size * current_size * current_channels)
            ),
            activation,
        )

    def forward(self, x):
        x = self.fc_layer(x)
        x = x.reshape((x.shape[0], self.conv_layers[0][0].in_channels, -1))
        x = x.reshape(
            (
                x.shape[0],
                self.conv_layers[0][0].in_channels,
                int(sqrt(x.shape[-1])),
                int(sqrt(x.shape[-1])),
            )
        )
        x = self.conv_layers(x)
        return x


class LinearEncoder(_BaseEncoder):
    def __init__(self, latent_dims: int, feature_size: int, variational: bool = False):
        super(LinearEncoder, self).__init__(latent_dims, variational=variational)
        self.variational = variational

        if self.variational:
            self.fc_mu = torch.nn.Linear(feature_size, latent_dims)
            self.fc_var = torch.nn.Linear(feature_size, latent_dims)
        else:
            self.fc = torch.nn.Linear(feature_size, latent_dims)

    def forward(self, x):
        if self.variational:
            mu = self.fc_mu(x)
            logvar = self.fc_var(x)
            return mu, logvar
        else:
            x = self.fc(x)
            return x


class LinearDecoder(_BaseDecoder):
    def __init__(self, latent_dims: int, feature_size: int):
        super(LinearDecoder, self).__init__(latent_dims)
        self.linear = torch.nn.Linear(latent_dims, feature_size)

    def forward(self, x):
        out = self.linear(x)
        return out
