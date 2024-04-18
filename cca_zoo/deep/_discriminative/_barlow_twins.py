from typing import List

import torch

from ._dcca import DCCA


class BarlowTwins(DCCA):
    """
    A class used to fit a Barlow Twins model.

    Barlow Twins is a self-supervised learning method that applies redundancy-reduction
    to learn representations that are invariant to distortions of the input sample.

    Parameters
    ----------

    lamb : float, optional
        off-diagonal scaling factor for the cross-covariance matrix. Defaults to 5e-3.

    References
    ----------
    Zbontar, Jure, et al. "Barlow twins: Self-supervised learning via redundancy reduction." arXiv preprint arXiv:2103.03230 (2021).

    """

    def __init__(
        self,
        *args,
        lamb=5e-3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lamb = lamb  # the lambda parameter for the off-diagonal terms of the cross-covariance matrix
        self.bns = torch.nn.ModuleList(
            [
                torch.nn.BatchNorm1d(self.latent_dimensions, affine=False)
                for _ in self.encoders
            ]
        )  # a list of batch normalization layers for each encoder

    def forward(self, views, **kwargs):
        representations = []
        for i, (encoder, bn) in enumerate(zip(self.encoders, self.bns)):
            representations.append(
                bn(encoder(views[i]))
            )  # encode and normalize each view
        return representations  # return a list of normalized latent representations

    def loss(
        self,
        representations: List[torch.Tensor],
        independent_representations: List[torch.Tensor] = None,
    ):
        cross_cov = (
            representations[0].T @ representations[1] / representations[0].shape[0]
        )  # compute the cross-covariance matrix between the two representations
        invariance = torch.sum(
            torch.pow(1 - torch.diag(cross_cov), 2)
        )  # compute the invariance term as the sum of squared differences from 1 on the diagonal
        covariance = torch.sum(
            torch.triu(torch.pow(cross_cov, 2), diagonal=1)
        ) + torch.sum(
            torch.tril(torch.pow(cross_cov, 2), diagonal=-1)
        )  # compute the covariance term as the sum of squared values on the off-diagonal
        return {
            "objective": invariance
            + self.lamb
            * covariance,  # return the objective value as a combination of invariance and covariance terms
            "invariance": invariance,  # return the invariance term
            "covariance": covariance,  # return the covariance term
        }
