from typing import List, Optional

import tensorly as tl
import torch
from tensorly.cp_tensor import cp_to_tensor
from tensorly.decomposition import parafac
from cca_zoo._utils.cross_correlation import torch_cross_cov


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


class _MCCALoss:
    """Differentiable MCCA Loss. Solves the multiset eigenvalue problem.

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
        C = C - torch.block_diag(
            *[torch.cov(representation.T) for representation in representations]
        )
        return C / len(representations)

    def D(self, representations):
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


class _GCCALoss:
    """Differentiable GCCA Loss. Solves the generalized CCA eigenproblem.

    References
    ----------
    https://arxiv.org/pdf/2005.11914.pdf
    """

    def __init__(self, eps: float = 1e-3):
        self.eps = eps

    def Q(self, representations):
        """Calculate Q matrix."""
        projections = [
            representation
            @ torch.linalg.inv(torch.cov(representation.T))
            @ representation.T
            for representation in representations
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
class _CCALoss:
    """Differentiable CCA Loss. Solves the CCA problem."""

    def __init__(self, eps: float = 1e-3):
        self.eps = eps

    def correlation(self, representations):
        """Calculate correlation."""
        latent_dims = representations[0].shape[1]
        o1 = representations[0].shape[1]
        o2 = representations[1].shape[1]

        representations = _demean(representations)

        SigmaHat12 = torch.cov(
            torch.hstack((representations[0], representations[1])).T
        )[:latent_dims, latent_dims:]
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


class _TCCALoss:
    """Differentiable TCCA Loss."""

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


@torch.jit.script
def CCA_AB(representations: list[torch.Tensor]):
    latent_dimensions = representations[0].shape[1]
    A = torch.zeros(
        latent_dimensions, latent_dimensions, device=representations[0].device
    )  # initialize the cross-covariance matrix
    B = torch.zeros(
        latent_dimensions, latent_dimensions, device=representations[0].device
    )  # initialize the auto-covariance matrix
    for zi in representations:
        B.add_(torch.cov(zi.T))  # In-place addition
        for zj in representations:
            A.add_(torch_cross_cov(zi, zj))  # In-place addition

    A.div_(len(representations))  # In-place division
    B.div_(len(representations))  # In-place division
    return A, B


@torch.jit.script
def PLS_AB(representations: list[torch.Tensor], weights: list[torch.Tensor]):
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
                B.add_(weights[i].T @ weights[i])  # In-place addition
            else:
                A.add_(torch_cross_cov(zi, zj))  # In-place addition

    A.div_(len(representations))  # In-place division
    B.div_(len(representations))  # In-place division
    return A, B


class _CCA_EYLoss:
    def __init__(self, eps: float = 1e-4):
        self.eps = eps

    @staticmethod
    @torch.jit.script
    def loss(
        representations: List[torch.Tensor],
        independent_representations: Optional[List[torch.Tensor]] = None,
    ):
        A, B = CCA_AB(representations)
        rewards = torch.trace(2 * A)
        if independent_representations is None:
            penalties = torch.trace(B @ B)
        else:
            independent_A, independent_B = CCA_AB(independent_representations)
            penalties = torch.trace(B @ independent_B)
        return {
            "objective": -rewards + penalties,
            "rewards": rewards,
            "penalties": penalties,
        }

    @staticmethod
    def derivative(
        views: List[torch.Tensor],
        representations: List[torch.Tensor],
        independent_views: Optional[List[torch.Tensor]] = None,
        independent_representations: Optional[List[torch.Tensor]] = None,
    ):
        A, B = CCA_AB(representations)
        sum_representations = torch.sum(torch.stack(representations), dim=0)
        n = sum_representations.shape[0]
        rewards = [2 * view.T @ sum_representations / (n - 1) for view in views]
        if independent_representations is None:
            penalties = [
                2 * view.T @ representation @ B / (n - 1)
                for view, representation in zip(views, representations)
            ]
        else:
            _, independent_B = CCA_AB(independent_representations)
            penalties = [
                view.T @ representation @ B / (n - 1)
                + independent_view.T
                @ independent_representation
                @ independent_B
                / (n - 1)
                for view, representation, independent_view, independent_representation in zip(
                    views,
                    representations,
                    independent_views,
                    independent_representations,
                )
            ]
        return [2 * (-reward + penalty) for reward, penalty in zip(rewards, penalties)]


class _CCA_GHALoss(_CCA_EYLoss):
    @staticmethod
    @torch.jit.script
    def loss(
        representations: List[torch.Tensor],
        independent_representations: Optional[List[torch.Tensor]] = None,
    ):
        A, B = CCA_AB(representations)
        rewards = torch.trace(A)
        if independent_representations is None:
            rewards += torch.trace(A)
            penalties = torch.trace(A @ B)
        else:
            independent_A, independent_B = CCA_AB(independent_representations)
            rewards += torch.trace(independent_A)
            penalties = torch.trace(independent_A @ B)
        return {
            "objective": -rewards + penalties,
            "rewards": rewards,
            "penalties": penalties,
        }

    @staticmethod
    def derivative(
        views: List[torch.Tensor],
        representations: List[torch.Tensor],
        independent_views: Optional[List[torch.Tensor]] = None,
        independent_representations: Optional[List[torch.Tensor]] = None,
    ):
        A, B = CCA_AB(representations)
        sum_representations = torch.sum(torch.stack(representations), dim=0)
        n = sum_representations.shape[0]
        if independent_representations is None:
            rewards = [2 * view.T @ sum_representations / (n - 1) for view in views]
            penalties = [
                view.T @ sum_representations @ B / (n - 1)
                + view.T @ representation @ A / (n - 1)
                for view, representation in zip(views, representations)
            ]
        else:
            independent_A, independent_B = CCA_AB(independent_representations)
            rewards = [2 * view.T @ sum_representations / (n - 1) for view in views]
            penalties = [
                view.T @ sum_representations @ independent_B / (n - 1)
                + view.T @ representation @ independent_A / (n - 1)
                for view, representation in zip(views, representations)
            ]
        return [2 * (-reward + penalty) for reward, penalty in zip(rewards, penalties)]


class _CCA_SVDLoss(_CCA_EYLoss):
    @staticmethod
    @torch.jit.script
    def loss(
        representations: List[torch.Tensor],
        independent_representations: Optional[List[torch.Tensor]] = None,
    ):
        C = torch.cov(torch.hstack(representations).T)
        latent_dims = representations[0].shape[1]

        Cxy = C[:latent_dims, latent_dims:]
        Cxx = C[:latent_dims, :latent_dims]

        if independent_representations is None:
            Cyy = C[latent_dims:, latent_dims:]
        else:
            Cyy = torch.cov(independent_representations[1].T)

        rewards = torch.trace(2 * Cxy)
        penalties = torch.trace(Cxx @ Cyy)
        return {
            "objective": -rewards + penalties,  # return the negative objective value
            "rewards": rewards,  # return the total rewards
            "penalties": penalties,  # return the penalties matrix
        }

    @staticmethod
    def derivative(
        views: List[torch.Tensor],
        representations: List[torch.Tensor],
        independent_views: Optional[List[torch.Tensor]] = None,
        independent_representations: Optional[List[torch.Tensor]] = None,
    ):
        C = torch.cov(torch.hstack(representations).T)
        latent_dims = representations[0].shape[1]

        Cxx = C[:latent_dims, :latent_dims]
        sum_representations = torch.sum(torch.stack(representations), dim=0)
        if independent_representations is None:
            Cyy = C[latent_dims:, latent_dims:]
        else:
            Cyy = torch.cov(independent_representations[1].T)
        n = sum_representations.shape[0]
        rewards = [
            2 * views[0].T @ representations[1] / (n - 1),
            2 * views[1].T @ representations[0] / (n - 1),
        ]
        penalties = [
            views[0].T @ representations[0] @ Cyy / (n - 1),
            views[1].T @ representations[1] @ Cxx / (n - 1),
        ]
        return [2 * (-reward + penalty) for reward, penalty in zip(rewards, penalties)]


class _PLS_EYLoss:
    @staticmethod
    @torch.jit.script
    def loss(representations: List[torch.Tensor], weights: List[torch.Tensor]):
        A, B = PLS_AB(representations, weights)
        rewards = torch.trace(2 * A)
        penalties = torch.trace(B @ B)
        return {
            "objective": -rewards + penalties,
            "rewards": rewards,
            "penalties": penalties,
        }

    @staticmethod
    @torch.jit.script
    def derivative(
        views: List[torch.Tensor],
        representations: List[torch.Tensor],
        weights: List[torch.Tensor],
    ):
        A, B = PLS_AB(representations, weights)
        sum_representations = torch.sum(torch.stack(representations), dim=0)
        n = sum_representations.shape[0]
        rewards = [2 * view.T @ sum_representations / (n - 1) for view in views]
        penalties = [
            2 * view.T @ representation @ B / (n - 1)
            for view, representation in zip(views, representations)
        ]
        return [2 * (-reward + penalty) for reward, penalty in zip(rewards, penalties)]


class _PLS_PowerLoss:
    @staticmethod
    @torch.jit.script
    def loss(representations: List[torch.Tensor]):
        cov = torch.cov(torch.hstack(representations).T)
        return {
            "objective": torch.trace(
                cov[: representations[0].shape[1], representations[0].shape[1] :]
            )
        }

    @staticmethod
    @torch.jit.script
    def derivative(views: List[torch.Tensor], representations: List[torch.Tensor]):
        grads = [views[0].T @ representations[1], views[1].T @ representations[0]]
        return grads
