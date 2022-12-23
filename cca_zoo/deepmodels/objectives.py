import tensorly as tl
import torch
from tensorly.cp_tensor import cp_to_tensor
from tensorly.decomposition import parafac


def _mat_pow(mat, pow_, epsilon):
    # Computing matrix to the power of pow (pow can be negative as well)
    [D, V] = torch.linalg.eigh(mat)
    mat_pow = V @ torch.diag((D + epsilon).pow(pow_)) @ V.T
    mat_pow[mat_pow != mat_pow] = epsilon  # For stability
    return mat_pow


def _demean(views):
    return tuple([view - view.mean(dim=0) for view in views])


class MCCA:
    """

    Differentiable MCCA Loss.
    Loss() method takes the outputs of each view's network and solves the multiset eigenvalue problem
    as in e.g. https://arxiv.org/pdf/2005.11914.pdf

    """

    def __init__(self, latent_dims: int, r: float = 0, eps: float = 1e-3):
        """

        :param latent_dims: the number of latent dimensions
        :param r: regularisation as in regularized CCA. Makes the problem well posed when batch size is similar to
        the number of latent dimensions
        :param eps: an epsilon parameter used in some operations
        """
        self.latent_dims = latent_dims
        self.r = r
        self.eps = eps

    def loss(self, views):
        n = views[0].shape[0]
        # Subtract the mean from each output
        views = _demean(views)

        # Concatenate all views and from this get the cross-covariance matrix
        all_views = torch.cat(views, dim=1)
        C = all_views.T @ all_views / (n - 1)

        # Get the block covariance matrix placing Xi^TX_i on the diagonal
        D = torch.block_diag(
            *[
                (1 - self.r) * m.T @ m / (n - 1)
                + self.r * torch.eye(m.shape[1], device=m.device)
                for i, m in enumerate(views)
            ]
        )

        C = C - torch.block_diag(*[view.T @ view / (n - 1) for view in views]) + D

        R = _mat_pow(D, -0.5, self.eps)

        # In MCCA our eigenvalue problem Cv = lambda Dv
        C_whitened = R @ C @ R.T

        eigvals = torch.linalg.eigvalsh(C_whitened)

        # Sort eigenvalues so lviewest first
        idx = torch.argsort(eigvals, descending=True)

        eigvals = eigvals[idx[: self.latent_dims]]

        # leaky relu encourages the gradient to be driven by positively correlated dimensions while also encouraging
        # dimensions associated with spurious negative correlations to become more positive
        eigvals = torch.nn.LeakyReLU()(eigvals[torch.gt(eigvals, 0)] - 1)

        corr = eigvals.sum()

        return -corr


class GCCA:
    """
    Differentiable GCCA Loss.
    Loss() method takes the outputs of each view's network and solves the generalized CCA eigenproblem
    as in https://arxiv.org/pdf/2005.11914.pdf

    """

    def __init__(self, latent_dims: int, r: float = 0, eps: float = 1e-3):
        """

        :param latent_dims: the number of latent dimensions
        :param r: regularisation as in regularized CCA. Makes the problem well posed when batch size is similar to
        the number of latent dimensions
        :param eps: an epsilon parameter used in some operations
        """
        self.latent_dims = latent_dims
        self.r = r
        self.eps = eps

    def loss(self, views):
        # https: // www.uta.edu / math / _docs / preprint / 2014 / rep2014_04.pdf
        n = views[0].shape[0]
        # H is n_views * n_samples * k
        views = _demean(views)

        eigen_views = [
            view @ _mat_pow(view.T @ view / (n - 1), -1, self.eps) @ view.T
            for view in views
        ]

        Q = torch.stack(eigen_views, dim=0).sum(dim=0)
        eigvals = torch.linalg.eigvalsh(Q)

        idx = torch.argsort(eigvals, descending=True)

        eigvals = eigvals[idx[: self.latent_dims]]

        # leaky relu encourages the gradient to be driven by positively correlated dimensions while also encouraging
        # dimensions associated with spurious negative correlations to become more positive
        eigvals = torch.nn.LeakyReLU()(eigvals[torch.gt(eigvals, 0)] - 1)

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
    """
    Differentiable CCA Loss.
    Loss() method takes the outputs of each view's network and solves the CCA problem as in Andrew's original paper

    """

    def __init__(self, latent_dims: int, r: float = 0, eps: float = 1e-3):
        """
        :param latent_dims: the number of latent dimensions
        :param r: regularisation as in regularized CCA. Makes the problem well posed when batch size is similar to the number of latent dimensions
        :param eps: an epsilon parameter used in some operations
        """
        self.latent_dims = latent_dims
        self.r = r
        self.eps = eps

    def loss(self, views):
        o1 = views[0].shape[1]
        o2 = views[1].shape[1]

        n = views[0].shape[0]

        views = _demean(views)

        SigmaHat12 = views[0].T @ views[1] / (n - 1)
        SigmaHat11 = (1 - self.r) / (n - 1) * views[0].T @ views[
            0
        ] + self.r * torch.eye(o1, device=views[0].device)
        SigmaHat22 = (1 - self.r) / (n - 1) * views[1].T @ views[
            1
        ] + self.r * torch.eye(o2, device=views[1].device)

        SigmaHat11RootInv = _mat_pow(SigmaHat11, -0.5, self.eps)
        SigmaHat22RootInv = _mat_pow(SigmaHat22, -0.5, self.eps)

        Tval = SigmaHat11RootInv @ SigmaHat12 @ SigmaHat22RootInv
        trace_TT = Tval.T @ Tval
        eigvals = torch.linalg.eigvalsh(trace_TT)

        # leaky relu encourages the gradient to be driven by positively correlated dimensions while also encouraging
        # dimensions associated with spurious negative correlations to become more positive
        eigvals = eigvals[torch.gt(eigvals, self.eps)]

        corr = torch.sum(torch.sqrt(eigvals))

        return -corr


class TCCA:
    """
    Differentiable TCCA Loss.

    """

    def __init__(self, latent_dims: int, r: float = 0, eps: float = 1e-4):
        """

        :param latent_dims: the number of latent dimensions
        :param r: regularisation as in regularized CCA. Makes the problem well posed when batch size is similar to the number of latent dimensions
        :param eps: an epsilon parameter used in some operations
        """
        self.latent_dims = latent_dims
        self.r = r
        self.eps = eps

    def loss(self, views):
        n = views[0].shape[0]
        views = _demean(views)
        covs = [
            (1 - self.r) * view.T @ view / (n - 1)
            + self.r * torch.eye(view.size(1), device=view.device)
            for view in views
        ]
        whitened_z = [
            view @ _mat_pow(cov, -0.5, self.eps) for view, cov in zip(views, covs)
        ]
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
            M.detach(), self.latent_dims, verbose=False, normalize_factors=True
        )
        M_parafac.weights = 1
        M_hat = cp_to_tensor(M_parafac)
        return torch.linalg.norm(M - M_hat)
