"""DGCCA — Deep Generalised CCA."""

from __future__ import annotations

import torch
import torch.nn as nn

from cca_zoo.deep._dcca import DCCA
from cca_zoo.deep.objectives import GCCALoss


class DGCCA(DCCA):
    r"""Deep Generalised CCA.

    Applies the generalised CCA (MAX-VAR) loss
    (:class:`~cca_zoo.deep.objectives.GCCALoss`) to neural representations,
    maximising correlation of each view with a shared latent target rather
    than with every other view pairwise:

    $$
    \mathcal{L} = -\sum_{d=1}^{k} \lambda_d\!\left(\sum_i H_i H_i^\top\right)
    $$

    where $H_i$ is the ridge-whitened representation of view
    $i$ and $\lambda_d(\cdot)$ is the $d$-th largest
    eigenvalue. Unlike the base :class:`DCCA`, this supports more than two
    encoders/views out of the box, mirroring the linear
    :class:`~cca_zoo.linear.GCCA`.

    References:
        Benton, A., et al. "Deep Generalized Canonical Correlation
        Analysis." RepL4NLP 2019.

    Args:
        latent_dimensions: Dimensionality of the shared latent space.
        encoders: List of :class:`torch.nn.Module` objects, one per view.
        objective: Ignored; the GCCA loss is always used. Accepted for
            API compatibility.
        lr: Learning rate. Default is 1e-3.
        max_epochs: Maximum training epochs. Default is 100.
        eps: Ridge regularisation for within-view whitening. Default is 1e-6.

    Example:
        >>> import torch.nn as nn
        >>> enc1 = nn.Linear(10, 4)
        >>> enc2 = nn.Linear(8, 4)
        >>> enc3 = nn.Linear(6, 4)
        >>> model = DGCCA(latent_dimensions=4, encoders=[enc1, enc2, enc3])
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
        # Override with GCCALoss regardless of what was passed
        self.objective = GCCALoss(eps=eps)

    def loss(
        self,
        representations: list[torch.Tensor],
        independent_representations: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the DGCCA loss via the generalised CCA objective.

        Args:
            representations: Encoded views from the current batch, each
                of shape (batch_size, latent_dimensions).
            independent_representations: Unused.

        Returns:
            Dictionary with key ``"objective"``.
        """
        return {"objective": self.objective(representations)}
