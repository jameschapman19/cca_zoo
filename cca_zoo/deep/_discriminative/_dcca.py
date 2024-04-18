from typing import List

import torch

from cca_zoo.deep._base import BaseDeep
from cca_zoo.deep._utils import inv_sqrtm


class DCCA(BaseDeep):
    """
    A class used to fit a DCCA model.

    References
    ----------

    """

    def loss(
        self,
        representations: List[torch.Tensor],
        independent_representations: List[torch.Tensor] = None,
    ):
        latent_dims = representations[0].shape[1]
        o1 = representations[0].shape[1]
        o2 = representations[1].shape[1]

        representations = [
            representation - representation.mean(dim=0)
            for representation in representations
        ]

        SigmaHat12 = torch.cov(
            torch.hstack((representations[0], representations[1])).T
        )[:latent_dims, latent_dims:]
        SigmaHat11 = torch.cov(representations[0].T) + 1e-5 * torch.eye(
            o1, device=representations[0].device
        )
        SigmaHat22 = torch.cov(representations[1].T) + 1e-5 * torch.eye(
            o2, device=representations[1].device
        )

        SigmaHat11RootInv = inv_sqrtm(SigmaHat11, 1e-5)
        SigmaHat22RootInv = inv_sqrtm(SigmaHat22, 1e-5)

        Tval = SigmaHat11RootInv @ SigmaHat12 @ SigmaHat22RootInv
        trace_TT = Tval.T @ Tval
        eigvals = torch.linalg.eigvalsh(trace_TT)
        eigvals = torch.nn.LeakyReLU()(eigvals[torch.gt(eigvals, 0)])
        return {"objective": -eigvals.sum()}
