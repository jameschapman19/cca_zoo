from typing import Iterable

import torch

from cca_zoo.deepmodels import objectives
from cca_zoo.deepmodels.architectures import BaseEncoder, Encoder
from cca_zoo.models import MCCA
from ._dcca_base import _DCCA_base


class DCCA(_DCCA_base, torch.nn.Module):
    """
    A class used to fit a DCCA model.

    Examples
    --------
    >>> from cca_zoo.deepmodels import DCCA
    >>> model = DCCA()
    """

    def __init__(self, latent_dims: int, objective=objectives.CCA,
                 encoders: Iterable[BaseEncoder] = [Encoder, Encoder],
                 learning_rate=1e-3, r: float = 1e-7, eps: float = 1e-7,
                 scheduler=None,
                 optimizer: torch.optim.Optimizer = None):
        """
        Constructor class for DCCA

        :param latent_dims: # latent dimensions
        :param objective: # CCA objective: normal tracenorm CCA by default
        :param encoders: list of encoder networks
        :param learning_rate: learning rate if no optimizer passed
        :param r: regularisation parameter of tracenorm CCA like ridge CCA. Needs to be VERY SMALL. If you get errors make this smaller
        :param eps: epsilon used throughout. Needs to be VERY SMALL. If you get errors make this smaller
        :param scheduler: scheduler associated with optimizer
        :param optimizer: pytorch optimizer
        """
        super().__init__(latent_dims)
        self.latent_dims = latent_dims
        self.encoders = torch.nn.ModuleList(encoders)
        self.objective = objective(latent_dims, r=r, eps=eps)
        if optimizer is None:
            # Andrew G, Arora R, Bilmes J, Livescu K. Deep canonical correlation analysis. InInternational conference on machine learning 2013 May 26 (pp. 1247-1255). PMLR.
            self.optimizer = torch.optim.LBFGS(self.parameters(), lr=learning_rate)
        else:
            self.optimizer = optimizer
        self.scheduler = scheduler
        self.eps = eps

    def update_weights(self, *args):
        if type(self.optimizer) == torch.optim.LBFGS:
            def closure():
                self.optimizer.zero_grad()
                z = self(*args)
                loss = self.objective.loss(*z)
                loss.backward()
                return loss

            self.optimizer.step(closure)
            loss = closure()
        else:
            self.optimizer.zero_grad()
            z = self(*args)
            loss = self.objective.loss(*z)
            loss.backward()
            self.optimizer.step()
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
