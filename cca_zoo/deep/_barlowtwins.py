"""BarlowTwins — Self-supervised learning via redundancy reduction (Zbontar 2021)."""

from __future__ import annotations

import torch
import torch.nn as nn

from cca_zoo.deep._dcca import DCCA


class BarlowTwins(DCCA):
    r"""Barlow Twins self-supervised learning model.

    Learns representations by encouraging the cross-correlation matrix
    between two views to be close to the identity:

    $$
    \mathcal{L} = \sum_i (1 - C_{ii})^2 + \lambda \sum_{i \neq j} C_{ij}^2,
    \qquad C = \frac{1}{n} Z_1^\top Z_2
    $$

    where $Z_1, Z_2$ are the batch-normalised representations of the
    two views. Batch normalisation is applied per-view before computing
    $C$.

    References:
        Zbontar, J., et al. "Barlow twins: Self-supervised learning via
        redundancy reduction." ICML 2021.

    Args:
        latent_dimensions: Dimensionality of the shared latent space.
        encoders: List of :class:`torch.nn.Module` objects, one per view.
        lam: Weight for the off-diagonal redundancy penalty.
            Default is 5e-3.
        objective: Ignored; the Barlow Twins loss is fixed. Accepted for
            API compatibility.
        lr: Learning rate. Default is 1e-3.
        max_epochs: Maximum training epochs. Default is 100.
        eps: Unused. Present for API compatibility. Default is 1e-6.

    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> enc1 = nn.Linear(10, 4)
        >>> enc2 = nn.Linear(8, 4)
        >>> model = BarlowTwins(latent_dimensions=4, encoders=[enc1, enc2], lam=5e-3)
    """

    def __init__(
        self,
        latent_dimensions: int,
        encoders: list[nn.Module],
        lam: float = 5e-3,
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
        self.lam = lam
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(latent_dimensions, affine=False) for _ in encoders]
        )

    def forward(self, views: list[torch.Tensor]) -> list[torch.Tensor]:
        """Encode views and apply batch normalisation.

        Args:
            views: List of input tensors, one per view.

        Returns:
            List of batch-normalised latent tensors.
        """
        return [bn(enc(v)) for enc, bn, v in zip(self.encoders, self.bns, views)]

    def loss(
        self,
        representations: list[torch.Tensor],
        independent_representations: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the Barlow Twins loss for two batch-normalised views.

        Args:
            representations: List containing exactly two batch-normalised
                tensors, each of shape (batch_size, latent_dimensions).
            independent_representations: Unused.

        Returns:
            Dictionary with keys ``"objective"``, ``"invariance"``, and
            ``"redundancy"``.
        """
        z1, z2 = representations[0], representations[1]
        n = z1.shape[0]
        cross_cov = z1.T @ z2 / n

        invariance = torch.sum(torch.pow(1.0 - torch.diag(cross_cov), 2))
        # Off-diagonal entries
        mask = ~torch.eye(cross_cov.shape[0], dtype=torch.bool, device=cross_cov.device)
        redundancy = torch.sum(torch.pow(cross_cov[mask], 2))
        objective = invariance + self.lam * redundancy
        return {
            "objective": objective,
            "invariance": invariance,
            "redundancy": redundancy,
        }
