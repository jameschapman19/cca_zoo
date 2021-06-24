from typing import Iterable
from typing import List

import numpy as np
import torch

from cca_zoo.deepmodels import objectives
from cca_zoo.deepmodels.architectures import BaseEncoder, Encoder
from cca_zoo.deepmodels.dcca import DCCA
from cca_zoo.models import MCCA


class DCCA_NOI(DCCA):
    """
    A class used to fit a DCCA model by non-linear orthogonal iterations

    Examples
    --------
    >>> from cca_zoo.deepmodels import DCCA_NOI
    >>> model = DCCA_NOI()
    """

    def __init__(self, latent_dims: int, N: int, objective=objectives.MCCA,
                 encoders: List[BaseEncoder] = [Encoder, Encoder],
                 learning_rate=1e-3, r: float = 1e-3, rho: float = 0.2, eps: float = 1e-3, shared_target: bool = False,
                 scheduler=None, optimizer: torch.optim.Optimizer = None):
        """
        Constructor class for DCCA

        :param latent_dims: # latent dimensions
        :param N: # samples used to estimate covariance
        :param objective: # CCA objective: normal tracenorm CCA by default
        :param encoders: list of encoder networks
        :param learning_rate: learning rate if no optimizer passed
        :param r: regularisation parameter of tracenorm CCA like ridge CCA
        :param rho: covariance memory like DCCA non-linear orthogonal iterations paper
        :param eps: epsilon used throughout
        :param shared_target: not used
        :param scheduler: scheduler associated with optimizer
        :param optimizer: pytorch optimizer
        """
        super().__init__(latent_dims=latent_dims, objective=objective, encoders=encoders, learning_rate=learning_rate,
                         r=r, eps=eps, scheduler=scheduler, optimizer=optimizer)
        self.N = N
        self.covs = None
        if rho < 0 or rho > 1:
            raise ValueError(f"rho should be between 0 and 1. rho={rho}")
        self.eps = eps
        self.rho = rho
        self.shared_target = shared_target

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
        self.update_covariances(*z)
        covariance_inv = [objectives._compute_matrix_power(objectives._minimal_regularisation(cov, self.eps), -0.5) for
                          cov in self.covs]
        preds = [torch.matmul(z, covariance_inv[i]).detach() for i, z in enumerate(z)]
        losses = [torch.mean(torch.norm(z_i - preds[-i], dim=0)) for i, z_i in enumerate(z, start=1)]
        loss = torch.sum(torch.stack(losses))
        return loss

    def update_covariances(self, *args):
        b = args[0].shape[0]
        batch_covs = [b / self.N * z_i.T @ z_i for i, z_i in enumerate(args)]
        if self.covs is not None:
            self.covs = [(self.rho * self.covs[i]).detach() + (1 - self.rho) * batch_cov for i, batch_cov
                         in
                         enumerate(batch_covs)]
        else:
            self.covs = batch_covs

    def post_transform(self, *z_list, train=False) -> Iterable[np.ndarray]:
        if train:
            self.cca = MCCA(latent_dims=self.latent_dims)
            self.cca.fit(*z_list)
            z_list = self.cca.transform(*z_list)
        else:
            z_list = self.cca.transform(*z_list)
        return z_list
