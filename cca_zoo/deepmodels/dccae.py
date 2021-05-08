from typing import Iterable

import torch
from torch.nn import functional as F

from cca_zoo.deepmodels.architectures import BaseEncoder, Encoder, BaseDecoder, Decoder
from cca_zoo.deepmodels.dcca import _DCCA_base
from cca_zoo.deepmodels.objectives import CCA


class DCCAE(_DCCA_base):
    """
    A class used to fit a DCCAE model.

    Examples
    --------
    >>> from cca_zoo.dccae import DCCAE
    >>> model = DCCAE()
    """

    def __init__(self, latent_dims: int, objective=CCA, encoders: Iterable[BaseEncoder] = [Encoder, Encoder],
                 decoders: Iterable[BaseDecoder] = [Decoder, Decoder], r: float = 1e-3, learning_rate=1e-3, lam=0.5,
                 schedulers: Iterable = None, optimizers: Iterable = None):
        """

        :param latent_dims: # latent dimensions
        :param objective: # CCA objective: normal tracenorm CCA by default
        :param encoders: list of encoder networks
        :param decoders:  list of decoder networks
        :param r:  regularisation parameter of tracenorm CCA like ridge CCA
        :param learning_rate: learning rate if no optimizers passed
        :param lam: weight of reconstruction loss (1 minus weight of correlation loss)
        :param schedulers: list of schedulers for each optimizer
        :param optimizers: list of optimizers for each encoder/decoder pair
        """
        super().__init__(latent_dims)
        self.encoders = torch.nn.ModuleList(encoders)
        self.decoders = torch.nn.ModuleList(decoders)
        self.lam = lam
        self.objective = objective(latent_dims, r=r)
        if optimizers is None:
            self.optimizers = [torch.optim.Adam(list(self.encoders.parameters()) + list(self.decoders.parameters()),
                                                lr=learning_rate)]
        else:
            self.optimizers = optimizers
        assert (0 <= self.lam <= 1), "lam between 0 and 1"
        self.schedulers = []
        if schedulers:
            self.schedulers.extend(schedulers)

    def update_weights(self, *args):
        [optimizer.zero_grad() for optimizer in self.optimizers]
        loss = self.loss(*args)
        loss.backward()
        [optimizer.step() for optimizer in self.optimizers]
        return loss

    def forward(self, *args):
        z = self.encode(*args)
        return z

    def encode(self, *args):
        z = []
        for i, encoder in enumerate(self.encoders):
            z.append(encoder(args[i]))
        return tuple(z)

    def decode(self, *args):
        recon = []
        for i, decoder in enumerate(self.decoders):
            recon.append(decoder(args[i]))
        return tuple(recon)

    def loss(self, *args):
        z = self.encode(*args)
        recon = self.decode(*z)
        recon_loss = self.recon_loss(args[:len(recon)], recon)
        return self.lam * recon_loss + self.objective.loss(*z)

    @staticmethod
    def recon_loss(x, recon):
        recons = [F.mse_loss(recon_, x_, reduction='sum') for recon_, x_ in zip(recon, x)]
        return torch.stack(recons).sum(dim=0)
