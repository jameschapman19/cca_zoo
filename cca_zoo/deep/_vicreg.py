"""VICReg — Variance-Invariance-Covariance Regularization (Bardes 2022)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cca_zoo.deep._dcca import DCCA


def _invariance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Compute the MSE similarity loss between two representations.

    Args:
        z1: Tensor of shape (batch_size, latent_dimensions).
        z2: Tensor of shape (batch_size, latent_dimensions).

    Returns:
        Scalar MSE loss.
    """
    return F.mse_loss(z1, z2)


def _variance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Compute the variance hinge loss to prevent collapsed representations.

    Penalises dimensions whose standard deviation falls below 1, encouraging
    each dimension to be used.

    Args:
        z1: Tensor of shape (batch_size, latent_dimensions).
        z2: Tensor of shape (batch_size, latent_dimensions).

    Returns:
        Scalar variance penalty.
    """
    eps = 1e-4
    std1 = torch.sqrt(z1.var(dim=0) + eps)
    std2 = torch.sqrt(z2.var(dim=0) + eps)
    return torch.mean(F.relu(1.0 - std1)) + torch.mean(F.relu(1.0 - std2))


def _covariance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Compute the off-diagonal covariance regularisation loss.

    Penalises correlation between different dimensions of the same view's
    representations to reduce redundancy.

    Args:
        z1: Tensor of shape (batch_size, latent_dimensions).
        z2: Tensor of shape (batch_size, latent_dimensions).

    Returns:
        Scalar covariance penalty.
    """
    n, d = z1.shape
    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)
    cov1 = (z1.T @ z1) / (n - 1)
    cov2 = (z2.T @ z2) / (n - 1)
    eye = torch.eye(d, device=z1.device, dtype=z1.dtype)
    off_diag_mask = ~eye.bool()
    penalty = (
        cov1[off_diag_mask].pow(2).sum() / d + cov2[off_diag_mask].pow(2).sum() / d
    )
    return penalty


class VICReg(DCCA):
    """Variance-Invariance-Covariance Regularization.

    Three-term self-supervised objective that jointly encourages::

        - Invariance: MSE similarity between the two views' representations.
        - Variance: Standard deviation of each feature dimension >= 1.
        - Covariance: Off-diagonal covariance close to zero.

    The total loss is::

        L = sim_coeff * MSE(z1, z2)
            + std_coeff * (hinge_var(z1) + hinge_var(z2))
            + cov_coeff * (off_diag_cov(z1) + off_diag_cov(z2))

    Reference:
        Bardes, A., Ponce, J., & LeCun, Y. "VICReg: Variance-Invariance-
        Covariance Regularization for Self-Supervised Learning."
        arXiv:2105.04906 (2022).

    Args:
        latent_dimensions: Dimensionality of the shared latent space.
        encoders: List of :class:`torch.nn.Module` objects, one per view.
        sim_coeff: Weight for the invariance (MSE) term. Default is 25.0.
        std_coeff: Weight for the variance hinge term. Default is 25.0.
        cov_coeff: Weight for the covariance penalty term. Default is 1.0.
        objective: Ignored; the VICReg loss is fixed. Accepted for API
            compatibility.
        lr: Learning rate. Default is 1e-3.
        max_epochs: Maximum training epochs. Default is 100.
        eps: Unused. Present for API compatibility. Default is 1e-6.

    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> enc1 = nn.Linear(10, 4)
        >>> enc2 = nn.Linear(8, 4)
        >>> model = VICReg(latent_dimensions=4, encoders=[enc1, enc2])
    """

    def __init__(
        self,
        latent_dimensions: int,
        encoders: list[nn.Module],
        sim_coeff: float = 25.0,
        std_coeff: float = 25.0,
        cov_coeff: float = 1.0,
        objective: nn.Module | None = None,
        lr: float = 1e-3,
        max_epochs: int = 100,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(
            latent_dimensions=latent_dimensions,
            encoders=encoders,
            objective=objective,
            lr=lr,
            max_epochs=max_epochs,
            eps=eps,
        )
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def loss(
        self,
        representations: list[torch.Tensor],
        independent_representations: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the VICReg three-term loss.

        Args:
            representations: List of tensors, each of shape
                (batch_size, latent_dimensions).  Currently only the
                first two views are used.
            independent_representations: Unused.

        Returns:
            Dictionary with keys ``"objective"``, ``"sim_loss"``,
            ``"var_loss"``, and ``"cov_loss"``.
        """
        z1, z2 = representations[0], representations[1]
        sim = _invariance_loss(z1, z2)
        var = _variance_loss(z1, z2)
        cov = _covariance_loss(z1, z2)
        objective = self.sim_coeff * sim + self.std_coeff * var + self.cov_coeff * cov
        return {
            "objective": objective,
            "sim_loss": sim,
            "var_loss": var,
            "cov_loss": cov,
        }
