from typing import Iterable

import torch

from cca_zoo.deepmodels import DCCA
from cca_zoo.deepmodels.architectures import BaseEncoder, Encoder


class BarlowTwins(DCCA):
    """
    A class used to fit a Barlow Twins model.

    :Citation:

    Zbontar, Jure, et al. "Barlow twins: Self-supervised learning via redundancy reduction." arXiv preprint arXiv:2103.03230 (2021).

    """

    def __init__(
        self,
        latent_dims: int,
        encoders: Iterable[BaseEncoder] = [Encoder, Encoder],
        lam=1,
    ):
        """
        Constructor class for Barlow Twins

        :param latent_dims: # latent dimensions
        :param encoders: list of encoder networks
        :param lam: weighting of off diagonal loss terms
        """
        super().__init__(latent_dims=latent_dims, encoders=encoders)
        self.lam = lam
        self.bns = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(latent_dims, affine=False) for _ in self.encoders]
        )

    def forward(self, *args):
        z = []
        for i, (encoder, bn) in enumerate(zip(self.encoders, self.bns)):
            z.append(bn(encoder(args[i])))
        return tuple(z)

    def loss(self, *args):
        z = self(*args)
        cross_cov = z[0].T @ z[1] / (z[0].shape[0] - 1)
        invariance = torch.mean(torch.pow(1 - torch.diag(cross_cov), 2))
        covariance = torch.mean(
            torch.triu(torch.pow(cross_cov, 2), diagonal=1)
        ) + torch.mean(torch.tril(torch.pow(cross_cov, 2), diagonal=-1))
        return invariance + covariance
