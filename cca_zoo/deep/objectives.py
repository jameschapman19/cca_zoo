import tensorly as tl
import torch
from tensorly.cp_tensor import cp_to_tensor
from tensorly.decomposition import parafac


def inv_sqrtm(A, eps=1e-9):
    """Compute the inverse square-root of a positive definite matrix."""
    # Perform eigendecomposition of covariance matrix
    U, S, V = torch.svd(A)
    # Enforce positive definite by taking a torch max() with eps
    S = torch.max(S, torch.tensor(eps, device=S.device))
    # Calculate inverse square-root
    inv_sqrt_S = torch.diag_embed(torch.pow(S, -0.5))
    # Calculate inverse square-root matrix
    B = torch.matmul(torch.matmul(U, inv_sqrt_S), V.transpose(-1, -2))
    return B


def _demean(views):
    return tuple([view - view.mean(dim=0) for view in views])


class MCCA:
    """Differentiable MCCA Loss. Solves the multiset eigenvalue problem.

    References
    ----------
    https://arxiv.org/pdf/2005.11914.pdf

    """

    def __init__(self, r: float = 0, eps: float = 1e-3):
        self.r = r
        self.eps = eps

    def C(self, views):
        """Calculate cross-covariance matrix."""
        all_views = torch.cat(views, dim=1)
        C = torch.cov(all_views.T)
        C = C - torch.block_diag(*[torch.cov(view.T) for view in views])
        return C / len(views)

    def D(self, views):
        """Calculate block covariance matrix."""
        D = torch.block_diag(
            *[
                (1 - self.r) * torch.cov(view.T)
                + self.r * torch.eye(view.shape[1], device=view.device)
                for view in views
            ]
        )
        return D / len(views)

    def correlation(self, views):
        """Calculate correlation."""
        latent_dims = views[0].shape[1]
        views = _demean(views)
        C = self.C(views)
        D = self.D(views)
        C += D
        R = inv_sqrtm(D, self.eps)
        C_whitened = R @ C @ R.T
        eigvals = torch.linalg.eigvalsh(C_whitened)
        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx[:latent_dims]]
        return eigvals

    def loss(self, views):
        """Calculate loss."""
        eigvals = self.correlation(views)
        eigvals = torch.nn.LeakyReLU()(eigvals[torch.gt(eigvals, 0)])
        corr = eigvals.sum()
        return -corr


class GCCA:
    """Differentiable GCCA Loss. Solves the generalized CCA eigenproblem.

    References
    ----------
    https://arxiv.org/pdf/2005.11914.pdf
    """

    def __init__(self, r: float = 0, eps: float = 1e-3):
        self.r = r
        self.eps = eps

    def Q(self, views):
        """Calculate Q matrix."""
        eigen_views = [
            view @ torch.linalg.inv(torch.cov(view.T)) @ view.T for view in views
        ]
        Q = torch.stack(eigen_views, dim=0).sum(dim=0)
        return Q

    def correlation(self, views):
        """Calculate correlation."""
        latent_dims = views[0].shape[1]
        views = _demean(views)
        Q = self.Q(views)
        eigvals = torch.linalg.eigvalsh(Q)
        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx[:latent_dims]]
        return torch.nn.LeakyReLU()(eigvals)

    def loss(self, views):
        """Calculate loss."""
        eigvals = self.correlation(views)
        corr = eigvals.sum()
        return -corr


# Original work Copyright (c) 2016 Vahid Noroozi
# Modified work Copyright 2019 Zhanghao Wu

# Permission is hereby granted, free of chviewe, to any person obtaining a copy
# of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:


# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
class CCA:
    """Differentiable CCA Loss. Solves the CCA problem."""

    def __init__(self, r: float = 0, eps: float = 1e-3):
        self.r = r
        self.eps = eps

    def correlation(self, views):
        """Calculate correlation."""
        latent_dims = views[0].shape[1]
        o1 = views[0].shape[1]
        o2 = views[1].shape[1]

        views = _demean(views)

        SigmaHat12 = torch.cov(torch.hstack((views[0], views[1])).T)[
            :latent_dims, latent_dims:
        ]
        SigmaHat11 = torch.cov(views[0].T) + self.r * torch.eye(
            o1, device=views[0].device
        )
        SigmaHat22 = torch.cov(views[1].T) + self.r * torch.eye(
            o2, device=views[1].device
        )

        SigmaHat11RootInv = inv_sqrtm(SigmaHat11, self.eps)
        SigmaHat22RootInv = inv_sqrtm(SigmaHat22, self.eps)

        Tval = SigmaHat11RootInv @ SigmaHat12 @ SigmaHat22RootInv
        trace_TT = Tval.T @ Tval
        eigvals = torch.linalg.eigvalsh(trace_TT)

        return eigvals

    def loss(self, views):
        """Calculate loss."""
        eigvals = self.correlation(views)
        eigvals = torch.nn.LeakyReLU()(eigvals[torch.gt(eigvals, 0)])
        return -eigvals.sum()


class TCCA:
    """Differentiable TCCA Loss."""

    def __init__(self, r: float = 0, eps: float = 1e-4):
        self.r = r
        self.eps = eps

    def loss(self, views):
        latent_dims = views[0].shape[1]
        views = _demean(views)
        covs = [
            (1 - self.r) * torch.cov(view.T)
            + self.r * torch.eye(view.size(1), device=view.device)
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
            # For the remaining views we expand their dimensions to match M i.e. nx1x...x1xp
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
