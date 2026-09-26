"""Barlow Twins."""

from __future__ import annotations

import torch
import torch.nn as nn

from cca_zoo.deep._base import BaseDeep, Batch, _require_two_views


class BarlowTwins(BaseDeep):
    r"""Barlow Twins: drive the cross-correlation of two views to the identity.

    $$
    \mathcal{L} = \sum_i (1 - C_{ii})^2 + \lambda \sum_{i \neq j} C_{ij}^2,
    \qquad C = \tfrac{1}{n} Z_1^\top Z_2,
    $$

    for batch-normalised encodings $Z_1, Z_2$.

    Args:
        n_components: Latent dimension.
        encoders: One module per view.
        lam: Weight of the off-diagonal term. Default is 5e-3.
        learning_rate: Adam learning rate. Default is 1e-3.

    Raises:
        ValueError: If there are not two encoders.

    References:
        Zbontar, J., Jing, L., Misra, I., LeCun, Y., & Deny, S. (2021).
        Barlow twins: Self-supervised learning via redundancy reduction.
        ICML.

    Examples:
        >>> import torch.nn as nn
        >>> from cca_zoo.deep import BarlowTwins
        >>> encoders = [nn.Linear(10, 4), nn.Linear(8, 4)]
        >>> model = BarlowTwins(n_components=4, encoders=encoders)
    """

    def __init__(
        self,
        n_components: int,
        encoders: list[nn.Module],
        lam: float = 5e-3,
        learning_rate: float = 1e-3,
    ) -> None:
        _require_two_views(encoders, "BarlowTwins")
        super().__init__(
            n_components=n_components,
            encoders=encoders,
            learning_rate=learning_rate,
        )
        self.lam = lam
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(n_components, affine=False) for _ in encoders]
        )

    def forward(self, views: list[torch.Tensor]) -> list[torch.Tensor]:
        """Batch-normalised encodings of each view."""
        return [bn(enc(v)) for enc, bn, v in zip(self.encoders, self.bns, views)]

    def loss(self, batch: Batch) -> dict[str, torch.Tensor]:
        """The Barlow Twins loss of a batch and its terms.

        Args:
            batch: Dictionary with a ``"views"`` list of tensors.

        Returns:
            ``{"objective", "invariance", "redundancy"}``.
        """
        representations = self(batch["views"])
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
