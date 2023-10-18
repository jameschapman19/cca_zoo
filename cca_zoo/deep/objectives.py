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


class MCCALoss:
    """Differentiable MCCALoss Loss. Solves the multiset eigenvalue problem.

    References
    ----------
    https://arxiv.org/pdf/2005.11914.pdf

    """

    def __init__(self, eps: float = 1e-3):
        self.eps = eps

    def C(self, representations):
        """Calculate cross-covariance matrix."""
        all_views = torch.cat(representations, dim=1)
        C = torch.cov(all_views.T)
        C = C - torch.block_diag(*[torch.cov(representation.T) for representation in representations])
        return C / len(representations)

    def D(self, representations):
        """Calculate block covariance matrix."""
        D = torch.block_diag(
            *[
                (1 - self.eps) * torch.cov(representation.T)
                + self.eps * torch.eye(representation.shape[1], device=representation.device)
                for representation in representations
            ]
        )
        return D / len(representations)

    def correlation(self, representations):
        """Calculate correlation."""
        latent_dims = representations[0].shape[1]
        representations = _demean(representations)
        C = self.C(representations)
        D = self.D(representations)
        C += D
        R = inv_sqrtm(D, self.eps)
        C_whitened = R @ C @ R.T
        eigvals = torch.linalg.eigvalsh(C_whitened)
        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx[:latent_dims]]
        return eigvals

    def loss(self, representations):
        """Calculate loss."""
        eigvals = self.correlation(representations)
        eigvals = torch.nn.LeakyReLU()(eigvals[torch.gt(eigvals, 0)])
        corr = eigvals.sum()
        return -corr


class GCCALoss:
    """Differentiable GCCALoss Loss. Solves the generalized CCALoss eigenproblem.

    References
    ----------
    https://arxiv.org/pdf/2005.11914.pdf
    """

    def __init__(self, eps: float = 1e-3):
        self.eps = eps

    def Q(self, representations):
        """Calculate Q matrix."""
        projections = [
            representation @ torch.linalg.inv(torch.cov(representation.T)) @ representation.T for representation in representations
        ]
        Q = torch.stack(projections, dim=0).sum(dim=0)
        return Q

    def correlation(self, representations):
        """Calculate correlation."""
        latent_dims = representations[0].shape[1]
        representations = _demean(representations)
        Q = self.Q(representations)
        eigvals = torch.linalg.eigvalsh(Q)
        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx[:latent_dims]]
        return torch.nn.LeakyReLU()(eigvals)

    def loss(self, representations):
        """Calculate loss."""
        eigvals = self.correlation(representations)
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
class CCALoss:
    """Differentiable CCALoss Loss. Solves the CCALoss problem."""

    def __init__(self, eps: float = 1e-3):
        self.eps = eps

    def correlation(self, representations):
        """Calculate correlation."""
        latent_dims = representations[0].shape[1]
        o1 = representations[0].shape[1]
        o2 = representations[1].shape[1]

        representations = _demean(representations)

        SigmaHat12 = torch.cov(torch.hstack((representations[0], representations[1])).T)[
            :latent_dims, latent_dims:
        ]
        SigmaHat11 = torch.cov(representations[0].T) + self.eps * torch.eye(
            o1, device=representations[0].device
        )
        SigmaHat22 = torch.cov(representations[1].T) + self.eps * torch.eye(
            o2, device=representations[1].device
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


class TCCALoss:
    """Differentiable TCCALoss Loss."""

    def __init__(self, eps: float = 1e-4):
        self.eps = eps

    def loss(self, views):
        latent_dims = views[0].shape[1]
        views = _demean(views)
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

class CCA_EYLoss:
    def __init__(self, eps: float = 1e-4):
        self.eps = eps

    def loss(self, representations, independent_representations=None):
        A, B = self.get_AB(representations)
        rewards = torch.trace(2 * A)
        if independent_representations is None:
            penalties = torch.trace(B @ B)
        else:
            independent_A, independent_B = self.get_AB(independent_representations)
            penalties = torch.trace(B @ independent_B)
        return {
            "objective": -rewards + penalties,
            "rewards": rewards,
            "penalties": penalties,
        }

    def get_AB(self, representations):
        latent_dimensions = representations[0].shape[1]
        A = torch.zeros(
            latent_dimensions, latent_dimensions, device=representations[0].device
        )  # initialize the cross-covariance matrix
        B = torch.zeros(
            latent_dimensions, latent_dimensions, device=representations[0].device
        )  # initialize the auto-covariance matrix
        for i, zi in enumerate(representations):
            for j, zj in enumerate(representations):
                if i == j:
                    B += torch.cov(zi.T)  # add the auto-covariance of each view to B
                else:
                    A += torch.cov(torch.hstack((zi, zj)).T)[
                         latent_dimensions :, : latent_dimensions
                         ]  # add the cross-covariance of each pair of representations to A
        return A / len(representations), B / len(
            representations
        )  # return the normalized matrices (divided by the number of representations)

class CCA_GHALoss(CCA_EYLoss):
    def loss(self, representations, independent_representations=None):
        A, B = self.get_AB(representations)
        rewards = torch.trace(2 * A)
        if independent_representations is None:
            penalties = torch.trace(B @ B)
        else:
            independent_A, independent_B = self.get_AB(independent_representations)
            penalties = torch.trace(B @ independent_B)
        return {
            "objective": -rewards + penalties,
            "rewards": rewards,
            "penalties": penalties,
        }

class CCA_SVDLoss(CCA_EYLoss):
    def loss(self, representations, independent_representations=None):
        C = torch.cov(torch.hstack(representations).T)
        latent_dims = representations[0].shape[1]

        Cxy = C[:latent_dims, latent_dims:]
        Cxx = C[:latent_dims, :latent_dims]

        if independent_representations is None:
            Cyy = C[latent_dims:, latent_dims:]
        else:
            Cyy = torch.cov(torch.hstack(independent_representations).T)[latent_dims:, latent_dims:]

        rewards = torch.trace(2 * Cxy)
        penalties = torch.trace(Cxx @ Cyy)
        return {
            "objective": -rewards + penalties,  # return the negative objective value
            "rewards": rewards,  # return the total rewards
            "penalties": penalties,  # return the penalties matrix
        }


class PLS_EYLoss(CCA_EYLoss):
    def loss(self, representations, weights=None):
        A, B = self.get_AB(representations, weights)
        rewards = torch.trace(2 * A)
        penalties = torch.trace(B @ B)
        return {
            "objective": -rewards + penalties,
            "rewards": rewards,
            "penalties": penalties,
        }
    def get_AB(self, representations, weights=None):
        latent_dimensions = representations[0].shape[1]
        A = torch.zeros(
            latent_dimensions, latent_dimensions, device=representations[0].device
        )  # initialize the cross-covariance matrix
        B = torch.zeros(
            latent_dimensions, latent_dimensions, device=representations[0].device
        )  # initialize the auto-covariance matrix
        n = representations[0].shape[0]
        for i, zi in enumerate(representations):
            for j, zj in enumerate(representations):
                if i == j:
                    B += weights[i].T @ weights[i] / n
                else:
                    A += torch.cov(torch.hstack((zi, zj)).T)[
                        latent_dimensions:, :latent_dimensions
                    ]  # add the cross-covariance of each pair of representations to A
        return A / len(representations), B / len(representations)


class PLS_SVDLoss(PLS_EYLoss):
    def loss(self, representations, weights=None):
        C = torch.cov(torch.hstack(representations).T)
        latent_dims = representations[0].shape[1]

        n = representations[0].shape[0]
        Cxy = C[:latent_dims, latent_dims:]
        Cxx = weights[0].T @ weights[0] / n
        Cyy = weights[1].T @ weights[1] / n

        rewards = torch.trace(2 * Cxy)
        penalties = torch.trace(Cxx @ Cyy)

        return {
            "objective": -rewards + penalties,
            "rewards": rewards,
            "penalties": penalties,
        }