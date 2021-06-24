from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from cca_zoo.deepmodels import objectives
from cca_zoo.deepmodels.architectures import BaseEncoder, Encoder, BaseDecoder, Decoder
from cca_zoo.deepmodels.dcca import _DCCA_base


class DCCAE(nn.Module, _DCCA_base):
    """
    A class used to fit a DCCAE model.

    Examples
    --------
    >>> from cca_zoo.deepmodels import DCCAE
    >>> model = DCCAE()
    """

    def __init__(self, latent_dims: int, objective=objectives.MCCA,
                 encoders: Iterable[BaseEncoder] = [Encoder, Encoder],
                 decoders: Iterable[BaseDecoder] = [Decoder, Decoder], r: float = 1e-7, eps: float = 1e-7,
                 learning_rate=1e-3, lam=0.5,
                 scheduler=None, optimizer: torch.optim.Optimizer = None):
        """
        :param latent_dims: # latent dimensions
        :param objective: # CCA objective: normal tracenorm CCA by default
        :param encoders: list of encoder networks
        :param decoders:  list of decoder networks
        :param r: regularisation parameter of tracenorm CCA like ridge CCA. Needs to be VERY SMALL. If you get errors make this smaller
        :param eps: epsilon used throughout. Needs to be VERY SMALL. If you get errors make this smaller
        :param learning_rate: learning rate if no optimizer passed
        :param lam: weight of reconstruction loss (1 minus weight of correlation loss)
        :param scheduler: scheduler associated with optimizer
        :param optimizer: pytorch optimizer
        """
        super(DCCAE, self).__init__()
        self.encoders = torch.nn.ModuleList(encoders)
        self.decoders = torch.nn.ModuleList(decoders)
        if lam < 0 or lam > 1:
            raise ValueError(f"lam should be between 0 and 1. rho={lam}")
        self.lam = lam
        self.objective = objective(latent_dims, r=r, eps=eps)
        if optimizer is None:
            # Wang W, Arora R, Livescu K, Bilmes J. On deep multi-view representation learning. InInternational conference on machine learning 2015 Jun 1 (pp. 1083-1092). PMLR.
            optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = scheduler
        _DCCA_base.__init__(self, latent_dims=latent_dims, optimizer=optimizer, scheduler=scheduler)

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
