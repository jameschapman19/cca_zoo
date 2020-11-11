import torch


def block_diag(m):
    """
    Make a block diagonal matrix along dim=-3
    EXAMPLE:
    block_diag(torch.ones(4,3,2))
    should give a 12 x 8 matrix with blocks of 3 x 2 ones.
    Prepend batch dimensions if needed.
    You can also give a list of matrices.
    :type m: torch.Tensor, list
    :rtype: torch.Tensor
    """
    if type(m) is list:
        m = torch.cat([m1.unsqueeze(-3) for m1 in m], -3)

    d = m.dim()
    n = m.shape[-3]
    siz0 = m.shape[:-3]
    siz1 = m.shape[-2:]
    m2 = m.unsqueeze(-2)
    eye = attach_dim(torch.eye(n, device=m.device).unsqueeze(-2), d - 3, 1)
    return (m2 * eye).reshape(
        siz0 + torch.Size(torch.tensor(siz1) * n)
    )


def attach_dim(v, n_dim_to_prepend=0, n_dim_to_append=0):
    return v.reshape(
        torch.Size([1] * n_dim_to_prepend)
        + v.shape
        + torch.Size([1] * n_dim_to_append))


class mcca:
    """
    Differentiable MCCA Loss.
    Loss() method takes the outputs of each view's network and solves the multiset eigenvalue problem
    """

    def __init__(self, outdim_size: int, r=1e-3, eps=1e-9):
        self.outdim_size = outdim_size
        self.r = r
        self.eps = eps

    def loss(self, *args):
        # https: // www.uta.edu / math / _docs / preprint / 2014 / rep2014_04.pdf
        # H is n_views * n_samples * k

        # Subtract the mean from each output
        views = [view - view.mean(dim=0) for view in args]

        # Concatenate all views and from this get the cross-covariance matrix
        all_views = torch.cat(views, dim=1)
        C = torch.matmul(all_views.T, all_views)

        # Get the block covariance matrix placing Xi^TX_i on the diagonal
        D = block_diag([torch.matmul(view.T, view) for view in views])

        # In MCCA our eigenvalue problem Cv = lambda Dv

        # Use the cholesky method to whiten the matrix C R^{-1}CRv = lambda v
        R = torch.cholesky(D, upper=True)

        C_whitened = torch.inverse(R.T) @ C @ torch.inverse(R)

        [eigvals, eigvecs] = torch.symeig(C_whitened, eigenvectors=True)

        # Sort eigenvalues so largest first
        idx = torch.argsort(eigvals, descending=True)

        # Sum the first #outdim_size values (after subtracting 1).
        corr = (eigvals[idx][:self.outdim_size] - 1).sum()

        return -corr


class gcca:
    """
    Differentiable GCCA Loss.
    Loss() method takes the outputs of each view's network and solves the multiset eigenvalue problem
    """

    def __init__(self, outdim_size: int, r=1e-3, eps=1e-9):
        self.outdim_size = outdim_size
        self.r = r
        self.eps = eps

    def loss(self, *args):
        # https: // www.uta.edu / math / _docs / preprint / 2014 / rep2014_04.pdf
        # H is n_views * n_samples * k
        all_views = [view - view.mean(dim=0) for view in args]

        eigen_views = [view @ torch.inverse(view.T @ view) @ view.T for view in all_views]

        Q = torch.stack(eigen_views, dim=0).sum(dim=0)

        [eigvals, eigvecs] = torch.symeig(Q, eigenvectors=True)

        idx = torch.argsort(eigvals, descending=True)
        eigvecs = eigvecs[:, idx]

        corr = (eigvals[idx][:self.outdim_size] - 1).sum()

        return -corr


class cca:
    """
    Differentiable CCA Loss.
    Loss() method takes the outputs of each view's network and solves the multiset eigenvalue problem
    """

    def __init__(self, outdim_size: int, r=1e-3, eps=1e-9):
        self.outdim_size = outdim_size
        self.r = r
        self.eps = eps

    def loss(self, H1, H2):
        H1, H2 = H1.t(), H2.t()

        o1 = o2 = H1.size(0)

        m = H1.size(1)

        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)

        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar,
                                                    H1bar.t()) + self.r * torch.eye(o1, dtype=torch.double,
                                                                                    device=H1.device)
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar,
                                                    H2bar.t()) + self.r * torch.eye(o2, dtype=torch.double,
                                                                                    device=H2.device)

        [D1, V1] = torch.symeig(SigmaHat11, eigenvectors=True)
        [D2, V2] = torch.symeig(SigmaHat22, eigenvectors=True)

        # Added to increase stability
        posInd1 = torch.gt(D1, self.eps).nonzero()[:, 0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = torch.gt(D2, self.eps).nonzero()[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]

        SigmaHat11RootInv = torch.matmul(
            torch.matmul(V1, torch.diag(torch.pow(D1, -0.5))), V1.t())
        SigmaHat22RootInv = torch.matmul(
            torch.matmul(V2, torch.diag(torch.pow(D2, -0.5))), V2.t())

        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv,
                                         SigmaHat12), SigmaHat22RootInv)

        # just the top self.outdim_size singular values are used
        trace_TT = torch.matmul(Tval.t(), Tval)
        # ensures positive definite
        U, V = torch.symeig(trace_TT, eigenvectors=True)
        U_inds = torch.gt(D1, self.eps).nonzero()[:, 0]
        U = U[U_inds]
        corr = torch.sum(torch.sqrt(U))
        return -corr
