"""DCCA — Deep Canonical Correlation Analysis (Andrew 2013)."""

from __future__ import annotations

import torch
import torch.nn as nn

from cca_zoo.deep._base import BaseDeep
from cca_zoo.deep.objectives import CCALoss


class DCCA(BaseDeep):
    r"""Deep Canonical Correlation Analysis with a pluggable objective.

    Trains two (or more) neural network encoders to maximise canonical
    correlation between their outputs, by minimising, per mini-batch:

    $$
    \mathcal{L} = -\left\|
        \Sigma_{11}^{-1/2} \Sigma_{12} \Sigma_{22}^{-1/2}
    \right\|_F^2
    $$

    where $\Sigma_{11}, \Sigma_{22}$ are the (ridge-regularised)
    within-view covariances of the two encoder outputs and
    $\Sigma_{12}$ their cross-covariance (see
    :class:`~cca_zoo.deep.objectives.CCALoss`). The objective is pluggable
    via the ``objective`` parameter, which defaults to the above.

    The model is a :class:`lightning.pytorch.LightningModule` and is
    trained via a :class:`lightning.Trainer`.

    References:
        Andrew, G., et al. "Deep canonical correlation analysis."
        ICML 2013.

    Args:
        latent_dimensions: Dimensionality of the shared latent space.
        encoders: List of :class:`torch.nn.Module` objects mapping each
            view to the latent space.
        objective: Differentiable loss module operating on a list of
            latent tensors.  If ``None``, defaults to
            :class:`~cca_zoo.deep.objectives.CCALoss`.
        lr: Learning rate for the Adam optimiser. Default is 1e-3.
        max_epochs: Maximum training epochs. Default is 100.
        eps: Regularisation parameter passed to the default CCALoss when
            ``objective`` is ``None``. Default is 1e-6.

    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> enc1 = nn.Linear(10, 4)
        >>> enc2 = nn.Linear(8, 4)
        >>> model = DCCA(latent_dimensions=4, encoders=[enc1, enc2])
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
        super().__init__(
            latent_dimensions=latent_dimensions,
            encoders=encoders,
            lr=lr,
            max_epochs=max_epochs,
            eps=eps,
        )
        self.objective: nn.Module = CCALoss(eps=eps) if objective is None else objective

    def loss(
        self,
        representations: list[torch.Tensor],
        independent_representations: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the DCCA training objective.

        Args:
            representations: Encoded views from the current batch, each
                of shape (batch_size, latent_dimensions).
            independent_representations: Optional second set of encodings
                (unused in the base DCCA formulation).

        Returns:
            Dictionary with key ``"objective"`` containing the scalar
            loss tensor to minimise.
        """
        return {"objective": self.objective(representations)}
