from typing import Iterable

import torch
import torch.nn.functional as F

from cca_zoo.deepmodels.architectures import BaseEncoder, Encoder, BaseDecoder, Decoder
from cca_zoo.deepmodels.dcca import _DCCA_base


class SplitAE(_DCCA_base):
    """
    A class used to fit a Split Autoencoder model.

    Examples
    --------
    >>> from cca_zoo.deepmodels import SplitAE
    >>> model = SplitAE()
    """

    def __init__(self, latent_dims: int, encoder: BaseEncoder = Encoder,
                 decoders: Iterable[BaseDecoder] = [Decoder, Decoder], learning_rate=1e-3,
                 scheduler=None, optimizer: torch.optim.Optimizer = None):
        """

        :param latent_dims: # latent dimensions
        :param encoder: list of encoder networks
        :param decoders:  list of decoder networks
        :param learning_rate: learning rate if no optimizers passed
        :param scheduler: scheduler associated with optimizer
        :param optimizer: pytorch optimizer

        """
        super().__init__(latent_dims)
        self.encoder = encoder
        self.decoders = torch.nn.ModuleList(decoders)
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(),
                                              lr=learning_rate)
        else:
            self.optimizer = optimizer
        self.scheduler = scheduler

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
