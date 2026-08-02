"""Differentiable CCA loss functions for use with deep models."""

from __future__ import annotations

import torch
import torch.nn as nn


def _inv_sqrtm(A: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Compute the inverse square root of a symmetric positive definite matrix.

    Args:
        A: Symmetric PD tensor of shape (n, n).
        eps: Regularisation added to eigenvalues for numerical stability.

    Returns:
        Tensor of shape (n, n): A^{-1/2}.
    """
    L, V = torch.linalg.eigh(A)
    L = torch.clamp(L, min=eps)
    return V @ torch.diag(1.0 / torch.sqrt(L)) @ V.T


class CCALoss(nn.Module):
    r"""Andrew 2013 deep CCA correlation loss for two views.

    Computes the negative sum of squared singular values of the whitened
    cross-covariance:

    $$
    \mathcal{L} = -\left\|
        \Sigma_{11}^{-1/2} \Sigma_{12} \Sigma_{22}^{-1/2}
    \right\|_F^2
    $$

    where $\Sigma_{11}, \Sigma_{22}$ are the (ridge-regularised)
    empirical within-view covariances of the two encoder outputs over the
    mini-batch and $\Sigma_{12}$ their cross-covariance. Minimising
    this loss maximises the sum of squared canonical correlations.

    References:
        Andrew, G., et al. "Deep canonical correlation analysis."
        ICML 2013.

    Args:
        eps: Ridge regularisation added to within-view covariance
            matrices for numerical stability. Default is 1e-5.

    Example:
        >>> import torch
        >>> loss_fn = CCALoss(eps=1e-4)
        >>> z1 = torch.randn(32, 4)
        >>> z2 = torch.randn(32, 4)
        >>> loss = loss_fn([z1, z2])
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, representations: list[torch.Tensor]) -> torch.Tensor:
        """Compute the CCA loss for a list containing exactly two views.

        Args:
            representations: List of two tensors, each of shape
                (batch_size, latent_dimensions).

        Returns:
            Scalar tensor: negative sum of squared canonical correlations.

        Raises:
            ValueError: If the number of representations is not exactly 2.
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
    r"""Multiview extension of CCALoss that sums pairwise CCA losses.

    $$
    \mathcal{L} = \sum_{i < j} \mathcal{L}_{\text{CCA}}(z_i, z_j)
    $$

    where $\mathcal{L}_{\text{CCA}}$ is :class:`CCALoss` applied to
    each pair of views. This is the deep, gradient-descent analogue of the
    SUMCOR multiset objective (Kettenring 1971) also used by the linear
    :class:`~cca_zoo.linear.MCCA`, and encourages every pair of views to be
    mutually correlated in the shared latent space.

    References:
        Kettenring, J. R. (1971). Canonical analysis of several sets of
        variables. *Biometrika*, 58(3), 433-451.

    Args:
        eps: Ridge regularisation passed to each pairwise CCALoss.
            Default is 1e-5.

    Example:
        >>> import torch
        >>> loss_fn = MCCALoss(eps=1e-4)
        >>> views = [torch.randn(32, 4) for _ in range(3)]
        >>> loss = loss_fn(views)
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self._cca_loss = CCALoss(eps=eps)

    def forward(self, representations: list[torch.Tensor]) -> torch.Tensor:
        """Compute the sum of pairwise CCA losses across all view pairs.

        Args:
            representations: List of tensors, each of shape
                (batch_size, latent_dimensions).

        Returns:
            Scalar tensor: sum of pairwise negative canonical correlations.
        """
        n_views = len(representations)
        total = torch.tensor(0.0, device=representations[0].device)
        for i in range(n_views):
            for j in range(i + 1, n_views):
                total = total + self._cca_loss([representations[i], representations[j]])
        return total


class GCCALoss(nn.Module):
    r"""Generalised CCA loss for multiple views (MAX-VAR GCCA objective).

    Maximises the sum of squared correlations between each whitened view
    and a shared latent target. Equivalently, minimises the negative sum
    of the top $k$ eigenvalues of the summed whitened Gram matrix:

    $$
    \mathcal{L} = -\sum_{d=1}^{k} \lambda_d\!\left(\sum_i H_i H_i^\top\right)
    $$

    where $H_i = \tilde{Z}_i (\tilde{Z}_i^\top \tilde{Z}_i +
    \epsilon I)^{-1/2}$ is the (mean-centred, ridge-whitened) representation
    of view $i$, and $\lambda_d(\cdot)$ denotes the $d$-th
    largest eigenvalue. This mirrors the generalised eigenvalue problem
    solved in closed form by the linear :class:`~cca_zoo.linear.GCCA`.

    References:
        Benton, A., et al. "Deep Generalized Canonical Correlation
        Analysis." RepL4NLP 2019.

    Args:
        eps: Ridge regularisation for within-view covariance inversion.
            Default is 1e-5.

    Example:
        >>> import torch
        >>> loss_fn = GCCALoss(eps=1e-4)
        >>> views = [torch.randn(32, 4) for _ in range(3)]
        >>> loss = loss_fn(views)
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, representations: list[torch.Tensor]) -> torch.Tensor:
        """Compute the GCCA loss.

        Args:
            representations: List of tensors, each of shape
                (batch_size, latent_dimensions).

        Returns:
            Scalar tensor: negative total GCCA objective.
        """
        n = representations[0].shape[0]
        whitened = []
        for z in representations:
            z_c = z - z.mean(dim=0)
            cov = (z_c.T @ z_c) / (n - 1) + self.eps * torch.eye(
                z_c.shape[1], device=z_c.device, dtype=z_c.dtype
            )
            whitened.append(z_c @ _inv_sqrtm(cov, self.eps))

        # M = sum_i H_i H_i^T, shape (n, n)
        m = torch.zeros(n, n, device=representations[0].device)
        for h in whitened:
            m = m + h @ h.T

        # Objective is trace of top singular values of M
        eigvals = torch.linalg.eigvalsh(m)
        k = representations[0].shape[1]
        top_eigvals = eigvals[-k:]
        return -top_eigvals.sum()


class TCCALoss(nn.Module):
    r"""Tensor CCA loss (proxy via Frobenius norm of cross-moment tensor).

    Forms the higher-order cross-moment tensor of the whitened
    representations and returns the negative Frobenius norm as a
    differentiable proxy for the tensor CCA objective:

    $$
    M = \frac{1}{n} \sum_{s=1}^{n} H_1[s] \otimes H_2[s] \otimes
        \cdots \otimes H_V[s], \qquad
    \mathcal{L} = -\left\| M \right\|_F
    $$

    where $\otimes$ denotes the outer product and $H_i =
    \tilde{Z}_i (\tilde{Z}_i^\top \tilde{Z}_i + \epsilon I)^{-1/2}$ is the
    whitened representation of view $i$. This is the deep,
    gradient-descent analogue of the higher-order cross-moment maximised
    in closed form by the linear :class:`~cca_zoo.linear.TCCA`.

    References:
        Kim, T.-K., Wong, S.-F., & Cipolla, R. (2007). Tensor canonical
        correlation analysis for action classification. *CVPR 2007*. IEEE.

    Args:
        eps: Ridge regularisation for whitening. Default is 1e-5.

    Example:
        >>> import torch
        >>> loss_fn = TCCALoss(eps=1e-4)
        >>> views = [torch.randn(32, 4) for _ in range(3)]
        >>> loss = loss_fn(views)
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, representations: list[torch.Tensor]) -> torch.Tensor:
        """Compute the tensor CCA loss.

        Args:
            representations: List of tensors, each of shape
                (batch_size, latent_dimensions).

        Returns:
            Scalar tensor: negative Frobenius norm of the cross-moment tensor.
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
        return -torch.linalg.norm(m)
