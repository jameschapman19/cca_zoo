from typing import Iterable

import torch
import torch.nn.functional as F

from cca_zoo.dcca import _DCCA_base
from cca_zoo.deep_models import BaseEncoder, Encoder, BaseDecoder, Decoder


class SplitAE(_DCCA_base):
    """
    A class used to fit a Split Autoencoder model.

    Examples
    --------
    >>> from cca_zoo.splitae import SplitAE
    >>> model = SplitAE()
    """

    def __init__(self, latent_dims: int, encoder: BaseEncoder = Encoder,
                 decoders: Iterable[BaseDecoder] = [Decoder, Decoder], learning_rate=1e-3,
                 schedulers: Iterable = None, optimizers: Iterable = None):
        """

        :param latent_dims: # latent dimensions
        :param encoders: list of encoder networks
        :param decoders:  list of decoder networks
        :param learning_rate: learning rate if no optimizers passed
        :param schedulers: list of schedulers for each optimizer
        :param optimizers: list of optimizers for each encoder
        """
        super().__init__(latent_dims)
        self.encoder = encoder
        self.decoders = torch.nn.ModuleList(decoders)
        if optimizers is None:
            self.optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoders.parameters()),
                                              lr=learning_rate)
        else:
            self.optimizer = optimizers
        self.schedulers = []
        if schedulers:
            self.schedulers.extend(schedulers)
        self.optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoders.parameters()),
                                          lr=learning_rate)

    def update_weights(self, *args):
        self.optimizer.zero_grad()
        loss = self.loss(*args)
        loss.backward()
        self.optimizer.step()
        return loss

    def forward(self, *args):
        z = self.encode(*args)
        return z

    def encode(self, *args):
        z = self.encoder(args[0])
        return z

    def decode(self, z):
        recon = []
        for i, decoder in enumerate(self.decoders):
            recon.append(decoder(z))
        return tuple(recon)

    def loss(self, *args):
        z = self.encode(*args)
        recon = self.decode(z)
        recon_loss = self.recon_loss(args, recon)
        return recon_loss

    @staticmethod
    def recon_loss(x, recon):
        recons = [F.mse_loss(recon[i], x[i], reduction='sum') for i in range(len(x))]
        return torch.stack(recons).sum(dim=0)
