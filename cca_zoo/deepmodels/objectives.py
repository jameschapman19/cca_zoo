import tensorly as tl
import torch
from tensorly.cp_tensor import cp_to_tensor
from tensorly.decomposition import parafac


def _minimal_regularisation(M, eps):
    # calculate smallest ammount of regularisation that ensures smallest eigenvalue is eps
    M_smallest_eig = torch.relu(-torch.min(torch.real(torch.linalg.eigvals(M)))) + eps
    M = M + M_smallest_eig * torch.eye(M.shape[0], dtype=torch.double, device=M.device).float()
    return M


def _compute_matrix_power(M, p):
    # torch.linalg.eig can be unstable if eigenvalues are the same or are small https://pytorch.org/docs/stable/generated/torch.linalg.eig.html
    U, V = torch.linalg.eig(M)
    M_p = torch.matmul(torch.matmul(torch.real(V), torch.diag(torch.pow(torch.real(U), p))), torch.real(V).t())
    return M_p


class MCCA:
    """

    Differentiable MCCA Loss.
    Loss() method takes the outputs of each view's network and solves the multiset eigenvalue problem
    as in e.g. https://arxiv.org/pdf/2005.11914.pdf

    """

    def __init__(self, latent_dims: int, r: float = 1e-7, eps: float = 1e-7):
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
        R = torch.linalg.cholesky(D)

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

    def __init__(self, latent_dims: int, r: float = 1e-7, eps: float = 1e-7):
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

        eigen_views = [view @ torch.inverse(_minimal_regularisation(view.T @ view, self.eps)) @ view.T for view in
                       all_views]

        Q = torch.stack(eigen_views, dim=0).sum(dim=0)

        [eigvals, eigvecs] = torch.linalg.eig(Q)

        idx = torch.argsort(eigvals.real, descending=True)
        eigvecs = eigvecs.real[:, idx]

        corr = (eigvals.real[idx][:self.latent_dims] - 1).sum()

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

    def __init__(self, latent_dims: int, r: float = 1e-7, eps: float = 1e-7):
        """
        :param latent_dims: the number of latent dimensions
        :param r: regularisation as in regularized CCA. Makes the problem well posed when batch size is similar to the number of latent dimensions
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
        SigmaHat11 = (1 - self.r) * (1.0 / (m - 1)) * torch.matmul(H1bar,
                                                                   H1bar.t()) + self.r * torch.eye(o1,
                                                                                                   dtype=torch.double,
                                                                                                   device=H1.device).float()
        SigmaHat22 = (1 - self.r) * (1.0 / (m - 1)) * torch.matmul(H2bar,
                                                                   H2bar.t()) + self.r * torch.eye(o2,
                                                                                                   dtype=torch.double,
                                                                                                   device=H2.device).float()

        SigmaHat11RootInv = _compute_matrix_power(_minimal_regularisation(SigmaHat11, self.eps), -0.5)
        SigmaHat22RootInv = _compute_matrix_power(_minimal_regularisation(SigmaHat22, self.eps), -0.5)

        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv,
                                         SigmaHat12), SigmaHat22RootInv)

        trace_TT = torch.matmul(Tval.t(), Tval)
        eigvals = torch.real(torch.linalg.eigvals(_minimal_regularisation(trace_TT, self.eps)))
        eigvals = eigvals[torch.gt(eigvals, self.eps)]
        corr = torch.sum(torch.sqrt(eigvals))
        return -corr


class TCCA:
    """
    Differentiable TCCA Loss.

    """

    def __init__(self, latent_dims: int, r: float = 1e-7, eps: float = 1e-7):
        """

        :param latent_dims: the number of latent dimensions
        :param r: regularisation as in regularized CCA. Makes the problem well posed when batch size is similar to the number of latent dimensions
        :param eps: an epsilon parameter used in some operations
        """
        self.latent_dims = latent_dims
        self.r = r
        self.eps = eps

    def loss(self, *z):
        m = z[0].size(0)
        z = [z_ - z_.mean(dim=0).unsqueeze(dim=0) for z_ in z]
        covs = [
            (1 - self.r) * (1.0 / (m - 1)) * torch.matmul(z_.T, z_) + self.r * torch.eye(z_.size(1), dtype=torch.double,
                                                                                         device=z_.device).float() for
            z_ in z]
        z = [z_ @ _compute_matrix_power(_minimal_regularisation(cov, self.eps), -0.5) for z_, cov in zip(z, covs)]

        for i, el in enumerate(z):
            if i == 0:
                curr = el
            else:
                for _ in range(len(curr.size()) - 1):
                    el = torch.unsqueeze(el, 1)
                curr = torch.unsqueeze(curr, -1) @ el
        M = torch.mean(curr, 0)
        tl.set_backend('pytorch')
        M_parafac = parafac(M.detach(), self.latent_dims)
        M_hat = cp_to_tensor(M_parafac)
        return torch.norm(M - M_hat)
