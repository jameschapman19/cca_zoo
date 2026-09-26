"""Variance-invariance-covariance regularisation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cca_zoo.deep._dcca import DCCA


def _invariance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Mean squared difference between two representations."""
    return F.mse_loss(z1, z2)


def _variance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Hinge penalty on each dimension's standard deviation falling below 1."""
    eps = 1e-4
    std1 = torch.sqrt(z1.var(dim=0) + eps)
    std2 = torch.sqrt(z2.var(dim=0) + eps)
    return torch.mean(F.relu(1.0 - std1)) + torch.mean(F.relu(1.0 - std2))


def _covariance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Squared off-diagonal covariances of each representation, over the dimension."""
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
    r"""Variance-invariance-covariance regularisation for two views.

    $$
    \mathcal{L} = \gamma \operatorname{MSE}(z_1, z_2)
        + \mu \sum_k \operatorname{mean}(\max(0, 1 - \sigma(z_k)))
        + \nu \sum_k \frac{1}{d} \sum_{i \neq j} \operatorname{Cov}(z_k)_{ij}^2,
    $$

    with $\gamma, \mu, \nu$ = ``sim_coeff``, ``std_coeff``, ``cov_coeff``.

    Args:
        n_components: Latent dimension.
        encoders: One module per view.
        sim_coeff: Weight of the invariance term. Default is 25.0.
        std_coeff: Weight of the variance term. Default is 25.0.
        cov_coeff: Weight of the covariance term. Default is 1.0.
        learning_rate: Adam learning rate. Default is 1e-3.
        max_epochs: Maximum training epochs. Default is 100.

    References:
        Bardes, A., Ponce, J., & LeCun, Y. (2022). VICReg:
        Variance-Invariance-Covariance Regularization for Self-Supervised
        Learning. ICLR.

    Examples:
        >>> import torch.nn as nn
        >>> from cca_zoo.deep import VICReg
        >>> model = VICReg(n_components=4, encoders=[nn.Linear(10, 4), nn.Linear(8, 4)])
    """

    def __init__(
        self,
        n_components: int,
        encoders: list[nn.Module],
        sim_coeff: float = 25.0,
        std_coeff: float = 25.0,
        cov_coeff: float = 1.0,
        learning_rate: float = 1e-3,
        max_epochs: int = 100,
    ) -> None:
        super().__init__(
            n_components=n_components,
            encoders=encoders,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
        )
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def loss(
        self,
        representations: list[torch.Tensor],
        independent_representations: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """The VICReg loss of the first two views and its terms.

        Args:
            representations: One tensor of shape (batch_size, n_components) per
                view.
            independent_representations: Unused.

        Returns:
            ``{"objective", "sim_loss", "var_loss", "cov_loss"}``.
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
