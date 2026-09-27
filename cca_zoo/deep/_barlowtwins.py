"""Barlow Twins."""

from __future__ import annotations

import itertools

import torch
import torch.nn as nn

from cca_zoo.deep._base import BaseDeep, Batch


class BarlowTwins(BaseDeep):
    r"""Barlow Twins: drive the cross-correlation between views to the identity.

    $$
    \mathcal{L} = \sum_{a < b} \Bigl( \sum_i (1 - C^{ab}_{ii})^2
        + \lambda \sum_{i \neq j} (C^{ab}_{ij})^2 \Bigr),
    \qquad C^{ab} = \tfrac{1}{n} Z_a^\top Z_b,
    $$

    for batch-normalised encodings $Z_a$, summed over pairs of views; with two
    views this is the original loss.

    Args:
        n_components: Latent dimension.
        encoders: One module per view.
        lam: Weight of the off-diagonal term. Default is 5e-3.
        learning_rate: Adam learning rate. Default is 1e-3.

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
        n, k = representations[0].shape
        off_diagonal = ~torch.eye(k, dtype=torch.bool, device=representations[0].device)
        invariance = redundancy = torch.zeros((), device=representations[0].device)
        for z1, z2 in itertools.combinations(representations, 2):
            cross = z1.T @ z2 / n
            invariance = invariance + torch.sum((1.0 - torch.diag(cross)) ** 2)
            redundancy = redundancy + torch.sum(cross[off_diagonal] ** 2)
        objective = invariance + self.lam * redundancy
        return {
            "objective": objective,
            "invariance": invariance,
            "redundancy": redundancy,
        }
