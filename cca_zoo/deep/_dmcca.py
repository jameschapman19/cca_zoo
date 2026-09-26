"""DMCCA — Deep Multiset CCA."""

from __future__ import annotations

import torch
import torch.nn as nn

from cca_zoo.deep._dcca import DCCA
from cca_zoo.deep.objectives import MCCALoss


class DMCCA(DCCA):
    r"""Deep Multiset CCA.

    Applies the multiview pairwise-sum CCA loss
    (:class:`~cca_zoo.deep.objectives.MCCALoss`) to neural representations,
    encouraging every pair of views to be mutually correlated in the shared
    latent space:

    $$
    \mathcal{L} = \sum_{i < j} \mathcal{L}_{\text{CCA}}(z_i, z_j)
    $$

    Unlike the base :class:`DCCA`, this supports more than two
    encoders/views out of the box. This is the same SUMCOR multiset
    objective used by the linear :class:`~cca_zoo.linear.MCCA`, here
    optimised over neural encoder outputs by gradient descent rather than
    via eigendecomposition.

    References:
        Kettenring, J. R. (1971). Canonical analysis of several sets of
        variables. *Biometrika*, 58(3), 433-451.

    Args:
        n_components: Dimensionality of the shared latent space.
        encoders: List of :class:`torch.nn.Module` objects, one per view.
        learning_rate: Learning rate. Default is 1e-3.
        max_epochs: Maximum training epochs. Default is 100.
        eps: Ridge regularisation passed to each pairwise CCA loss.
            Default is 1e-6.

    Examples:
        >>> import torch.nn as nn
        >>> enc1 = nn.Linear(10, 4)
        >>> enc2 = nn.Linear(8, 4)
        >>> enc3 = nn.Linear(6, 4)
        >>> model = DMCCA(n_components=4, encoders=[enc1, enc2, enc3])
    """

    def __init__(
        self,
        n_components: int,
        encoders: list[nn.Module],
        learning_rate: float = 1e-3,
        max_epochs: int = 100,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(
            n_components=n_components,
            encoders=encoders,
            objective=MCCALoss(eps=eps),
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            eps=eps,
        )

    def loss(
        self,
        representations: list[torch.Tensor],
        independent_representations: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the DMCCA loss via the summed pairwise CCA objective.

        Args:
            representations: Encoded views from the current batch, each
                of shape (batch_size, n_components).
            independent_representations: Unused.

        Returns:
            Dictionary with key ``"objective"``.
        """
        return {"objective": self.objective(representations)}
