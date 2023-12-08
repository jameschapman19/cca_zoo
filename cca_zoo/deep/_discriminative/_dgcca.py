from typing import List

import torch

from ._dcca import DCCA


class _GCCALoss:
    """Differentiable GCCA Loss. Solves the generalized CCA eigenproblem.

    References
    ----------
    https://arxiv.org/pdf/2005.11914.pdf
    """

    def __init__(self, eps: float = 1e-3):
        self.eps = eps

    def Q(self, representations: List[torch.Tensor]):
        """Calculate Q matrix."""
        projections = [
            representation
            @ torch.linalg.inv(torch.cov(representation.T))
            @ representation.T
            for representation in representations
        ]
        Q = torch.stack(projections, dim=0).sum(dim=0)
        return Q

    def correlation(self, representations: List[torch.Tensor]):
        """Calculate correlation."""
        latent_dims = representations[0].shape[1]
        representations = [
            representation - representation.mean(dim=0)
            for representation in representations
        ]
        Q = self.Q(representations)
        eigvals = torch.linalg.eigvalsh(Q)
        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx[:latent_dims]]
        return torch.nn.LeakyReLU()(eigvals)

    def __call__(self, representations: List[torch.Tensor]):
        """Calculate loss."""
        eigvals = self.correlation(representations)
        corr = eigvals.sum()
        return -corr


class DGCCA(DCCA):
    """
    A class used to fit a DGCCA model.

    Is just a thin wrapper round DCCA with the DGCCA objective

    References
    ----------


    """

    objective = _GCCALoss()
