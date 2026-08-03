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
        latent_dimensions: Dimensionality of the shared latent space.
        encoders: List of :class:`torch.nn.Module` objects, one per view.
        objective: Ignored; the MCCA loss is always used. Accepted for
            API compatibility.
        lr: Learning rate. Default is 1e-3.
        max_epochs: Maximum training epochs. Default is 100.
        eps: Ridge regularisation passed to each pairwise CCA loss.
            Default is 1e-6.

    Example:
        >>> import torch.nn as nn
        >>> enc1 = nn.Linear(10, 4)
        >>> enc2 = nn.Linear(8, 4)
        >>> enc3 = nn.Linear(6, 4)
        >>> model = DMCCA(latent_dimensions=4, encoders=[enc1, enc2, enc3])
    """

    def __init__(
        self,
        latent_dimensions: int,
        encoders: list[nn.Module],
        objective: nn.Module | None = None,
        lr: float = 1e-3,
        max_epochs: int = 100,
        eps: float = 1e-6,
    ) -> None:
        # Pass objective=None so DCCA creates CCALoss, but we override it
        super().__init__(
            latent_dimensions=latent_dimensions,
            encoders=encoders,
            objective=None,
            lr=lr,
            max_epochs=max_epochs,
            eps=eps,
        )
        # Override with MCCALoss regardless of what was passed
        self.objective = MCCALoss(eps=eps)

    def loss(
        self,
        representations: list[torch.Tensor],
        independent_representations: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the DMCCA loss via the summed pairwise CCA objective.

        Args:
            representations: Encoded views from the current batch, each
                of shape (batch_size, latent_dimensions).
            independent_representations: Unused.

        Returns:
            Dictionary with key ``"objective"``.
        """
        return {"objective": self.objective(representations)}
