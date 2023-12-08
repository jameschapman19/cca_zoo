from typing import List

import torch

from ._dcca import DCCA
from .._utils import inv_sqrtm


class _MCCALoss:
    """Differentiable MCCA Loss. Solves the multiset eigenvalue problem.

    References
    ----------
    https://arxiv.org/pdf/2005.11914.pdf

    """

    def __init__(self, eps: float = 1e-3):
        self.eps = eps

    def C(self, representations: List[torch.Tensor]):
        """Calculate cross-covariance matrix."""
        all_views = torch.cat(representations, dim=1)
        C = torch.cov(all_views.T)
        C = C - torch.block_diag(
            *[torch.cov(representation.T) for representation in representations]
        )
        return C / len(representations)

    def D(self, representations: List[torch.Tensor]):
        """Calculate block covariance matrix."""
        D = torch.block_diag(
            *[
                (1 - self.eps) * torch.cov(representation.T)
                + self.eps
                * torch.eye(representation.shape[1], device=representation.device)
                for representation in representations
            ]
        )
        return D / len(representations)

    def correlation(self, representations: List[torch.Tensor]):
        """Calculate correlation."""
        latent_dims = representations[0].shape[1]
        representations = [
            representation - representation.mean(dim=0)
            for representation in representations
        ]
        C = self.C(representations)
        D = self.D(representations)
        C += D
        R = inv_sqrtm(D, self.eps)
        C_whitened = R @ C @ R.T
        eigvals = torch.linalg.eigvalsh(C_whitened)
        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx[:latent_dims]]
        return eigvals

    def __call__(self, representations: List[torch.Tensor]):
        """Calculate loss."""
        eigvals = self.correlation(representations)
        eigvals = torch.nn.LeakyReLU()(eigvals[torch.gt(eigvals, 0)])
        corr = eigvals.sum()
        return -corr


class DMCCA(DCCA):
    """
    A class used to fit a DMCCA model.

    Is just a thin wrapper round DCCA with the DMCCA objective

    References
    ----------


    """
    objective = _MCCALoss()
