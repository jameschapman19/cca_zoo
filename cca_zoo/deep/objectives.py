"""Differentiable CCA losses for the deep models."""

from __future__ import annotations

import torch
import torch.nn as nn


def _inv_sqrtm(A: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Inverse square root of a symmetric matrix, eigenvalues floored at ``eps``."""
    L, V = torch.linalg.eigh(A)
    L = torch.clamp(L, min=eps)
    inv_sqrt: torch.Tensor = V @ torch.diag(1.0 / torch.sqrt(L)) @ V.T
    return inv_sqrt


class CCALoss(nn.Module):
    r"""Two-view deep CCA loss (Andrew et al., 2013).

    $$
    \mathcal{L} = -\| \Sigma_{11}^{-1/2} \Sigma_{12} \Sigma_{22}^{-1/2} \|_F^2,
    $$

    minus the sum of squared canonical correlations of the batch, with
    ridge-regularised within-view covariances.

    Args:
        eps: Ridge added to the within-view covariances. Default is 1e-5.

    References:
        Andrew, G., Arora, R., Bilmes, J., & Livescu, K. (2013). Deep
        canonical correlation analysis. ICML.

    Examples:
        >>> import torch
        >>> from cca_zoo.deep.objectives import CCALoss
        >>> loss = CCALoss(eps=1e-4)([torch.randn(32, 4), torch.randn(32, 4)])
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, representations: list[torch.Tensor]) -> torch.Tensor:
        """The loss of two views.

        Args:
            representations: One tensor of shape (batch_size, n_components) per
                view.

        Returns:
            The loss.

        Raises:
            ValueError: If there are not exactly two views.
        """
        if len(representations) != 2:
            raise ValueError(
                "CCALoss expects exactly 2 representations, "
                f"got {len(representations)}."
            )
        z1, z2 = representations
        n = z1.shape[0]
        d1, d2 = z1.shape[1], z2.shape[1]

        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)

        s11 = (z1.T @ z1) / (n - 1) + self.eps * torch.eye(
            d1, device=z1.device, dtype=z1.dtype
        )
        s22 = (z2.T @ z2) / (n - 1) + self.eps * torch.eye(
            d2, device=z2.device, dtype=z2.dtype
        )
        s12 = (z1.T @ z2) / (n - 1)

        s11_inv_sqrt = _inv_sqrtm(s11, self.eps)
        s22_inv_sqrt = _inv_sqrtm(s22, self.eps)

        t = s11_inv_sqrt @ s12 @ s22_inv_sqrt
        # Squared singular values = eigenvalues of T^T T
        tt = t.T @ t
        eigvals = torch.linalg.eigvalsh(tt)
        eigvals = torch.clamp(eigvals, min=0.0)
        return -eigvals.sum()


class MCCALoss(nn.Module):
    """Sum of :class:`CCALoss` over every pair of views.

    Args:
        eps: Ridge of each pairwise loss. Default is 1e-5.

    References:
        Kettenring, J. R. (1971). Canonical analysis of several sets of
        variables. Biometrika, 58(3), 433-451.

    Examples:
        >>> import torch
        >>> from cca_zoo.deep.objectives import MCCALoss
        >>> loss = MCCALoss(eps=1e-4)([torch.randn(32, 4) for _ in range(3)])
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self._cca_loss = CCALoss(eps=eps)

    def forward(self, representations: list[torch.Tensor]) -> torch.Tensor:
        """The summed pairwise loss.

        Args:
            representations: One tensor of shape (batch_size, n_components) per
                view.

        Returns:
            The loss.
        """
        n_views = len(representations)
        total = torch.tensor(0.0, device=representations[0].device)
        for i in range(n_views):
            for j in range(i + 1, n_views):
                total = total + self._cca_loss([representations[i], representations[j]])
        return total


class GCCALoss(nn.Module):
    r"""Generalized CCA loss: minus the top eigenvalues of the summed projections.

    $$
    \mathcal{L} = -\sum_{d=1}^{k} \lambda_d\Bigl(\sum_i P_i\Bigr), \qquad
    P_i = \frac{H_i H_i^\top}{n - 1},
    $$

    for ridge-whitened, centred encodings $H_i$, so that $P_i$ projects onto
    view $i$'s encodings. The loss lies in $[-kM, 0]$ for $M$ views at any
    batch size, and is $-kM$ when every view's encodings agree.

    Args:
        eps: Whitening ridge. Default is 1e-5.

    References:
        Benton, A., Khayrallah, H., Gujral, B., Reisinger, D. A., Zhang, S.,
        & Arora, R. (2019). Deep generalized canonical correlation analysis.
        RepL4NLP.

    Examples:
        >>> import torch
        >>> from cca_zoo.deep.objectives import GCCALoss
        >>> loss = GCCALoss(eps=1e-4)([torch.randn(32, 4) for _ in range(3)])
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, representations: list[torch.Tensor]) -> torch.Tensor:
        """The generalized CCA loss.

        Args:
            representations: One tensor of shape (batch_size, n_components) per
                view.

        Returns:
            The loss.
        """
        n = representations[0].shape[0]
        whitened = []
        for z in representations:
            z_c = z - z.mean(dim=0)
            cov = (z_c.T @ z_c) / (n - 1) + self.eps * torch.eye(
                z_c.shape[1], device=z_c.device, dtype=z_c.dtype
            )
            whitened.append(z_c @ _inv_sqrtm(cov, self.eps))

        # The summed projections onto each view's encodings, shape (n, n).
        m = sum(h @ h.T for h in whitened) / (n - 1)
        eigvals = torch.linalg.eigvalsh(m)
        k = representations[0].shape[1]
        top_eigvals = eigvals[-k:]
        objective: torch.Tensor = -top_eigvals.sum()
        return objective


class TCCALoss(nn.Module):
    r"""Tensor CCA loss: minus the norm of the whitened cross-moment tensor.

    $$
    \mathcal{L} = -\| M \|_F, \qquad
    M = \tfrac{1}{n} \sum_s H_1[s] \otimes \cdots \otimes H_M[s],
    $$

    for ridge-whitened, centred encodings $H_i$.

    Args:
        eps: Whitening ridge. Default is 1e-5.

    References:
        Luo, Y., Tao, D., Ramamohanarao, K., Xu, C., & Wen, Y. (2015). Tensor
        canonical correlation analysis for multi-view dimension reduction.
        IEEE Transactions on Knowledge and Data Engineering, 27(11), 3111-3124.

    Examples:
        >>> import torch
        >>> from cca_zoo.deep.objectives import TCCALoss
        >>> loss = TCCALoss(eps=1e-4)([torch.randn(32, 4) for _ in range(3)])
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, representations: list[torch.Tensor]) -> torch.Tensor:
        """The tensor CCA loss.

        Args:
            representations: One tensor of shape (batch_size, n_components) per
                view.

        Returns:
            The loss.
        """
        n = representations[0].shape[0]
        whitened = []
        for z in representations:
            z_c = z - z.mean(dim=0)
            cov = (z_c.T @ z_c) / (n - 1) + self.eps * torch.eye(
                z_c.shape[1], device=z_c.device, dtype=z_c.dtype
            )
            whitened.append(z_c @ _inv_sqrtm(cov, self.eps))

        # Build outer product tensor iteratively, shape (d, d, ..., d)
        m: torch.Tensor = whitened[0]
        for i in range(1, len(whitened)):
            el = whitened[i]
            for _ in range(len(m.shape) - 1):
                el = el.unsqueeze(1)
            m = m.unsqueeze(-1) * el

        # Average over samples
        m = m.mean(dim=0)
        objective: torch.Tensor = -torch.linalg.norm(m)
        return objective
