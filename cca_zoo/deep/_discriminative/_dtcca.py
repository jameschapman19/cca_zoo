from typing import List

import tensorly as tl
import torch
from tensorly import cp_to_tensor
from tensorly.decomposition import parafac

from ._dcca import DCCA
from .._utils import inv_sqrtm
from ...linear._tcca import TCCA


class _TCCALoss:
    """Differentiable TCCA Loss."""

    def __init__(self, eps: float = 1e-4):
        self.eps = eps

    def __call__(self, representations: List[torch.Tensor]):
        latent_dims = representations[0].shape[1]
        views = [
            representation - representation.mean(dim=0)
            for representation in representations
        ]
        covs = [
            (1 - self.eps) * torch.cov(view.T)
            + self.eps * torch.eye(view.size(1), device=view.device)
            for view in views
        ]
        whitened_z = [view @ inv_sqrtm(cov, self.eps) for view, cov in zip(views, covs)]
        # The idea here is to form a matrix with M dimensions one for each view where at index
        # M[p_i,p_j,p_k...] we have the sum over n samples of the product of the pth feature of the
        # ith, jth, kth view etc.
        for i, el in enumerate(whitened_z):
            # To achieve this we start with the first view so M is nxp.
            if i == 0:
                M = el
            # For the remaining representations we expand their dimensions to match M i.e. nx1x...x1xp
            else:
                for _ in range(len(M.size()) - 1):
                    el = torch.unsqueeze(el, 1)
                # Then we perform an outer product by expanding the dimensionality of M and
                # outer product with the expanded el
                M = torch.unsqueeze(M, -1) @ el
        M = torch.mean(M, 0)
        tl.set_backend("pytorch")
        M_parafac = parafac(
            M.detach(), latent_dims, verbose=False, normalize_factors=True
        )
        M_parafac.weights = 1
        M_hat = cp_to_tensor(M_parafac)
        return torch.linalg.norm(M - M_hat)


class DTCCA(TCCA, DCCA):
    """
    A class used to fit a DTCCA model.

    Is just a thin wrapper round DCCA with the DTCCA objective

    References
    ----------
    Wong, Hok Shing, et al. "Deep Tensor CCA for Multi-view Learning." IEEE Transactions on Big Data (2021).

    """

    objective = _TCCALoss()