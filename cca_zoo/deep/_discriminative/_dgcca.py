from typing import List

import torch

from ._dcca import DCCA


class DGCCA(DCCA):
    """
    A class used to fit a DGCCA model.

    References
    ----------
    https://arxiv.org/pdf/2005.11914.pdf
    """

    def loss(
        self,
        representations: List[torch.Tensor],
        independent_representations: List[torch.Tensor] = None,
    ):
        latent_dims = representations[0].shape[1]
        representations = [
            representation - representation.mean(dim=0)
            for representation in representations
        ]
        Q = self.Q(representations)
        eigvals = torch.linalg.eigvalsh(Q)
        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx[:latent_dims]]
        eigvals = torch.nn.LeakyReLU()(eigvals)
        corr = eigvals.sum()
        return {"objective": -corr}

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
