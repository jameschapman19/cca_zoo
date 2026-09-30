"""Variance-invariance-covariance regularisation."""

from __future__ import annotations

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from cca_zoo.deep._base import BaseDeep, Batch


def _variance_loss(z: torch.Tensor) -> torch.Tensor:
    """Hinge penalty on each dimension's standard deviation falling below 1."""
    return torch.mean(F.relu(1.0 - torch.sqrt(z.var(dim=0) + 1e-4)))


def _covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """Squared off-diagonal covariances of a representation, over its dimension."""
    d = z.shape[1]
    cov = torch.cov(z.T)
    return cov[~torch.eye(d, dtype=torch.bool, device=z.device)].pow(2).sum() / d


class VICReg(BaseDeep):
    r"""Variance-invariance-covariance regularisation.

    $$
    \mathcal{L} = \gamma \sum_{a < b} \operatorname{MSE}(z_a, z_b)
        + \mu \sum_a \operatorname{mean}(\max(0, 1 - \sigma(z_a)))
        + \nu \sum_a \frac{1}{d} \sum_{i \neq j} \operatorname{Cov}(z_a)_{ij}^2,
    $$

    with $\gamma, \mu, \nu$ = ``sim_coeff``, ``std_coeff``, ``cov_coeff``. The
    invariance term is summed over pairs of views; with two views this is the
    original loss.

    Args:
        n_components: Number of latent dimensions.
        encoders: One module per view.
        sim_coeff: Weight of the invariance term. Default is 25.0.
        std_coeff: Weight of the variance term. Default is 25.0.
        cov_coeff: Weight of the covariance term. Default is 1.0.
        learning_rate: Adam learning rate. Default is 1e-3.

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
    ) -> None:
        super().__init__(
            n_components=n_components,
            encoders=encoders,
            learning_rate=learning_rate,
        )
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def loss(self, batch: Batch) -> dict[str, torch.Tensor]:
        """The VICReg loss of a batch and its terms.

        Args:
            batch: Dictionary with a ``"views"`` list of tensors.

        Returns:
            ``{"objective", "sim_loss", "var_loss", "cov_loss"}``.
        """
        representations = self(batch["views"])
        sim = torch.stack(
            [F.mse_loss(a, b) for a, b in itertools.combinations(representations, 2)]
        ).sum()
        var = torch.stack([_variance_loss(z) for z in representations]).sum()
        cov = torch.stack([_covariance_loss(z) for z in representations]).sum()
        objective = self.sim_coeff * sim + self.std_coeff * var + self.cov_coeff * cov
        return {
            "objective": objective,
            "sim_loss": sim,
            "var_loss": var,
            "cov_loss": cov,
        }
