from typing import List

import torch

from cca_zoo.deepmodels.architectures import BaseEncoder, Encoder
from cca_zoo.deepmodels.objectives import CCA
from cca_zoo.models.wrappers import MCCA
from ._dcca_base import _DCCA_base


class DCCA(_DCCA_base, torch.nn.Module):
    """
    A class used to fit a DCCA model.

    Examples
    --------
    >>> from cca_zoo.dcca import DCCA
    >>> model = DCCA()
    """

    def __init__(self, latent_dims: int, objective=CCA,
                 encoders: List[BaseEncoder] = [Encoder, Encoder],
                 learning_rate=1e-3, r: float = 1e-3, eps: float = 1e-9,
                 schedulers: List = None,
                 optimizers: List[torch.optim.Optimizer] = None):
        """
        Constructor class for DCCA

        :param latent_dims: # latent dimensions
        :param objective: # CCA objective: normal tracenorm CCA by default
        :param encoders: list of encoder networks
        :param learning_rate: learning rate if no optimizers passed
        :param r: regularisation parameter of tracenorm CCA like ridge CCA
        :param eps: epsilon used throughout
        :param schedulers: list of schedulers for each optimizer
        :param optimizers: list of optimizers for each encoder
        """
        super().__init__(latent_dims)
        self.latent_dims = latent_dims
        self.encoders = torch.nn.ModuleList(encoders)
        self.objective = objective(latent_dims, r=r)
        if optimizers is None:
            self.optimizers = [torch.optim.Adam(list(encoder.parameters()), lr=learning_rate) for encoder in
                               self.encoders]
        else:
            self.optimizers = optimizers
        self.schedulers = []
        if schedulers:
            self.schedulers.extend(schedulers)
        self.covs = None
        self.eps = eps

    def update_weights(self, *args):
        [optimizer.zero_grad() for optimizer in self.optimizers]
        z = self(*args)
        loss = self.objective.loss(*z)
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

    def loss(self, *args):
        z = self(*args)
        return self.objective.loss(*z)

    def post_transform(self, *z_list, train=False):
        if train:
            self.cca = MCCA(latent_dims=self.latent_dims)
            self.cca.fit(*z_list)
            z_list = self.cca.transform(*z_list)
        else:
            z_list = self.cca.transform(*z_list)
        return z_list
