"""DTCCA — Deep Tensor CCA (Wong 2021)."""

from __future__ import annotations

import torch
import torch.nn as nn

from cca_zoo.deep._dcca import DCCA
from cca_zoo.deep.objectives import TCCALoss


class DTCCA(DCCA):
    """Deep Tensor CCA.

    Applies the tensor CCA loss to neural representations.  The
    cross-moment tensor is formed from whitened latent codes, and the
    objective is the negative Frobenius norm of that tensor (serving as
    a differentiable proxy for the TCCA criterion).

    Reference:
        Wong, H. S., et al. "Deep Tensor CCA for Multi-view Learning."
        IEEE Transactions on Big Data (2021).

    Args:
        latent_dimensions: Dimensionality of the shared latent space.
        encoders: List of :class:`torch.nn.Module` objects, one per view.
        objective: Ignored; the TCCA loss is always used. Accepted for
            API compatibility.
        lr: Learning rate. Default is 1e-3.
        max_epochs: Maximum training epochs. Default is 100.
        eps: Ridge regularisation for whitening. Default is 1e-6.

    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> enc1 = nn.Linear(10, 4)
        >>> enc2 = nn.Linear(8, 4)
        >>> enc3 = nn.Linear(6, 4)
        >>> model = DTCCA(latent_dimensions=4, encoders=[enc1, enc2, enc3])
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
        # Override with TCCALoss regardless of what was passed
        self.objective = TCCALoss(eps=eps)

    def loss(
        self,
        representations: list[torch.Tensor],
        independent_representations: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the DTCCA loss via the tensor cross-moment.

        Args:
            representations: Encoded views from the current batch, each
                of shape (batch_size, latent_dimensions).
            independent_representations: Unused.

        Returns:
            Dictionary with key ``"objective"``.
        """
        return {"objective": self.objective(representations)}
