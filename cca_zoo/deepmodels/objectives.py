import numpy as np
import scipy.linalg
import tensorly as tl
import torch
from tensorly.cp_tensor import cp_to_tensor
from tensorly.decomposition import parafac
from torch.autograd import Function


class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.
    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
    """

    @staticmethod
    def forward(ctx, input):
        m = input.detach().cpu().numpy().astype(np.float_)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input

def _minimal_regularisation(M, eps):
    M_smallest_eig = torch.relu(-torch.min(torch.linalg.eigvalsh(M))) + eps
    M = M + M_smallest_eig * torch.eye(M.shape[0], device=M.device)
    return M

def _demean(*views):
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

    def loss(self, *views):
        # Subtract the mean from each output
        views = _demean(*views)

        # Concatenate all views and from this get the cross-covariance matrix
        all_views = torch.cat(views, dim=1)
        C = all_views.T @ all_views

        # Get the block covariance matrix placing Xi^TX_i on the diagonal
        D = torch.block_diag(
            *[(1 - self.r) * m.T @ m + self.r * torch.eye(m.shape[1], device=m.device) for i, m in enumerate(views)])

        C = C - torch.block_diag(*[view.T @ view for view in views]) + D

        D = _minimal_regularisation(D, self.eps)

        R = torch.linalg.inv(MatrixSquareRoot.apply(_minimal_regularisation(D, self.eps)))

        # In MCCA our eigenvalue problem Cv = lambda Dv
        C_whitened = R @ C @ R.T

        eigvals = torch.linalg.eigvalsh(C_whitened)

        # Sort eigenvalues so lviewest first
        idx = torch.argsort(eigvals, descending=True)

        eigvals = eigvals[idx[:self.latent_dims]]

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

    def loss(self, *views):
        # https: // www.uta.edu / math / _docs / preprint / 2014 / rep2014_04.pdf
        # H is n_views * n_samples * k
        views = _demean(*views)

        eigen_views = [view @ torch.inverse(_minimal_regularisation(view.T @ view, self.eps)) @ view.T for view in
                       views]

        Q = torch.stack(eigen_views, dim=0).sum(dim=0)
        eigvals = torch.linalg.eigvalsh(Q)

        idx = torch.argsort(eigvals, descending=True)

        eigvals = eigvals[idx[:self.latent_dims]]

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

    def loss(self, H1, H2):
        o1 = H1.shape[1]
        o2 = H2.shape[1]

        n = H1.shape[0]

        H1bar, H2bar = _demean(H1, H2)

        SigmaHat12 = (1.0 / (n - 1)) * H1bar.T @ H2bar
        SigmaHat11 = (1 - self.r) * (1.0 / (n - 1)) * H1bar.T @ H1bar + self.r * torch.eye(o1, device=H1.device)
        SigmaHat22 = (1 - self.r) * (1.0 / (n - 1)) * H2bar.T @ H2bar + self.r * torch.eye(o2, device=H2.device)

        SigmaHat11RootInv = torch.linalg.inv(MatrixSquareRoot.apply(_minimal_regularisation(SigmaHat11, self.eps)))
        SigmaHat22RootInv = torch.linalg.inv(MatrixSquareRoot.apply(_minimal_regularisation(SigmaHat22, self.eps)))

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

    def __init__(self, latent_dims: int, r: float = 0, eps: float = 1e-3):
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
            (1 - self.r) * (1.0 / (m - 1)) * z_.T @ z_ + self.r * torch.eye(z_.size(1), device=z_.device)
            for
            z_ in z]
        z = [z_ @ torch.linalg.inv(MatrixSquareRoot.apply(_minimal_regularisation(cov, self.eps))) for z_, cov in
             zip(z, covs)]

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