from typing import List

import torch

from ._dcca import DCCA
from .._utils import inv_sqrtm


class DMCCA(DCCA):
    """
    A class used to fit a DMCCA model.

    Is just a thin wrapper round DCCA with the DMCCA objective

    References
    ----------


    """

    def loss(
        self,
        representations: List[torch.Tensor],
        independent_representations: List[torch.Tensor]=None,
    ):
        latent_dims = representations[0].shape[1]
        representations = [
            representation - representation.mean(dim=0)
            for representation in representations
        ]
        A = self.A(representations)
        B = self.B(representations)
        A += B
        R = inv_sqrtm(B, self.eps)
        C_whitened = R @ A @ R.T
        eigvals = torch.linalg.eigvalsh(C_whitened)
        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx[:latent_dims]]
        eigvals = torch.nn.LeakyReLU()(eigvals[torch.gt(eigvals, 0)])
        corr = eigvals.sum()
        return {"objective":-corr}

    def A(self, representations: List[torch.Tensor]):
        """Calculate cross-covariance matrix."""
        all_views = torch.cat(representations, dim=1)
        A = torch.cov(all_views.T)
        A = A - torch.block_diag(
            *[torch.cov(representation.T) for representation in representations]
        )
        return A / len(representations)

    def B(self, representations: List[torch.Tensor]):
        """Calculate block covariance matrix."""
        B = torch.block_diag(
            *[
                (1 - self.eps) * torch.cov(representation.T)
                + self.eps
                * torch.eye(representation.shape[1], device=representation.device)
                for representation in representations
            ]
        )
        return B / len(representations)
