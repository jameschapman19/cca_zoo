import torch


def attach_dim(v, n_dim_to_prepend=0, n_dim_to_append=0):
    return v.reshape(
        torch.Size([1] * n_dim_to_prepend)
        + v.shape
        + torch.Size([1] * n_dim_to_append))


def compute_matrix_power(M, p, eps, order=True):
    [D, V] = torch.symeig(M, eigenvectors=True)
    if order:
        posInd1 = torch.nonzero(torch.gt(D, eps))[:, 0]
        D = D[posInd1]
        V = V[:, posInd1]
    M_p = torch.matmul(torch.matmul(V, torch.diag(torch.pow(D, p))), V.t())
    return M_p


class MCCA:
    """
    Differentiable MCCA Loss.
    Loss() method takes the outputs of each view's network and solves the multiset eigenvalue problem
    as in e.g. https://arxiv.org/pdf/2005.11914.pdf
    """

    def __init__(self, latent_dims: int, r: float = 1e-3, eps: float = 0):
        """
        :param latent_dims: the number of latent dimensions
        :param r: regularisation as in regularized CCA. Makes the problem well posed when batch size is similar to
        the number of latent dimensions
        :param eps: an epsilon parameter used in some operations
        """
        self.latent_dims = latent_dims
        self.r = r
        self.eps = eps

    def loss(self, *views):
        # https: // www.uta.edu / math / _docs / preprint / 2014 / rep2014_04.pdf
        # H is n_views * n_samples * k

        # Subtract the mean from each output
        views = [view - view.mean(dim=0) for view in views]

        # Concatenate all views and from this get the cross-covariance matrix
        all_views = torch.cat(views, dim=1)
        C = torch.matmul(all_views.T, all_views)

        # Get the block covariance matrix placing Xi^TX_i on the diagonal
        D = torch.block_diag(*[torch.matmul(view.T, view) for view in views])

        # In MCCA our eigenvalue problem Cv = lambda Dv

        # Use the cholesky method to whiten the matrix C R^{-1}CRv = lambda v
        R = torch.cholesky(D, upper=True)

        C_whitened = torch.inverse(R.T) @ C @ torch.inverse(R)

        [eigvals, eigvecs] = torch.symeig(C_whitened, eigenvectors=True)

        # Sort eigenvalues so lviewest first
        idx = torch.argsort(eigvals, descending=True)

        # Sum the first #latent_dims values (after subtracting 1).
        corr = (eigvals[idx][:self.latent_dims] - 1).sum()

        return -corr


class GCCA:
    """
    Differentiable GCCA Loss.
    Loss() method takes the outputs of each view's network and solves the generalized CCA eigenproblem
    as in https://arxiv.org/pdf/2005.11914.pdf
    """

    def __init__(self, latent_dims: int, r: float = 1e-3, eps: float = 0):
        """
        :param latent_dims: the number of latent dimensions
        :param r: regularisation as in regularized CCA. Makes the problem well posed when batch size is similar to
        the number of latent dimensions
        :param eps: an epsilon parameter used in some operations
        """
        self.latent_dims = latent_dims
        self.r = r
        self.eps = eps

    def loss(self, *views):
        # https: // www.uta.edu / math / _docs / preprint / 2014 / rep2014_04.pdf
        # H is n_views * n_samples * k
        all_views = [view - view.mean(dim=0) for view in views]

        eigen_views = [view @ torch.inverse(view.T @ view) @ view.T for view in all_views]

        Q = torch.stack(eigen_views, dim=0).sum(dim=0)

        [eigvals, eigvecs] = torch.symeig(Q, eigenvectors=True)

        idx = torch.argsort(eigvals, descending=True)
        eigvecs = eigvecs[:, idx]

        corr = (eigvals[idx][:self.latent_dims] - 1).sum()

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

    def __init__(self, latent_dims: int, r: float = 1e-3, eps: float = 0):
        """
        :param latent_dims: the number of latent dimensions
        :param r: regularisation as in regularized CCA. Makes the problem well posed when batch size is similar to
        the number of latent dimensions
        :param eps: an epsilon parameter used in some operations
        """
        self.latent_dims = latent_dims
        self.r = r
        self.eps = eps

    def loss(self, H1, H2):
        H1, H2 = H1.t(), H2.t()

        o1 = H1.size(0)
        o2 = H2.size(0)

        m = H1.size(1)

        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)

        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar,
                                                    H1bar.t()) + self.r * torch.eye(o1, dtype=torch.double,
                                                                                    device=H1.device).float()
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar,
                                                    H2bar.t()) + self.r * torch.eye(o2, dtype=torch.double,
                                                                                    device=H2.device).float()

        SigmaHat11RootInv = compute_matrix_power(SigmaHat11, -0.5, self.eps)
        SigmaHat22RootInv = compute_matrix_power(SigmaHat22, -0.5, self.eps)

        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv,
                                         SigmaHat12), SigmaHat22RootInv)

        # just the top self.latent_dims singular values are used
        trace_TT = torch.matmul(Tval.t(), Tval)
        U, V = torch.symeig(trace_TT, eigenvectors=True)
        U_inds = torch.nonzero(torch.gt(U, self.eps))[:, 0]
        U = U[U_inds]
        corr = torch.sum(torch.sqrt(U))
        return -corr


a = CCA(latent_dims=1, eps=0)
b = torch.ones(100, 1) + 0.0000001 * torch.rand(100, 1)
c = torch.rand(100, 1)
m = a.loss(b, c)
print('here')
