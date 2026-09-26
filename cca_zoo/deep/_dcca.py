"""Deep CCA."""

from __future__ import annotations

import torch
import torch.nn as nn

from cca_zoo.deep._base import BaseDeep
from cca_zoo.deep.objectives import CCALoss


class DCCA(BaseDeep):
    r"""Deep CCA: neural encoders trained to maximise canonical correlation.

    By default minimises, per mini-batch,

    $$
    \mathcal{L} = -\bigl\| \Sigma_{11}^{-1/2} \Sigma_{12} \Sigma_{22}^{-1/2} \bigr\|_F^2
    $$

    (:class:`~cca_zoo.deep.objectives.CCALoss`); pass ``objective`` to use
    another loss.

    Args:
        n_components: Latent dimension.
        encoders: One module per view.
        objective: Loss on the list of encodings; None uses ``CCALoss(eps)``.
            Default is None.
        learning_rate: Adam learning rate. Default is 1e-3.
        max_epochs: Maximum training epochs. Default is 100.
        eps: Ridge of the default loss. Default is 1e-6.

    References:
        Andrew, G., Arora, R., Bilmes, J., & Livescu, K. (2013). Deep
        canonical correlation analysis. ICML.

    Examples:
        >>> import torch.nn as nn
        >>> from cca_zoo.deep import DCCA
        >>> model = DCCA(n_components=4, encoders=[nn.Linear(10, 4), nn.Linear(8, 4)])
    """

    def __init__(
        self,
        n_components: int,
        encoders: list[nn.Module],
        objective: nn.Module | None = None,
        learning_rate: float = 1e-3,
        max_epochs: int = 100,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(
            n_components=n_components,
            encoders=encoders,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
        )
        self.eps = eps
        self.objective: nn.Module = CCALoss(eps=eps) if objective is None else objective

    def loss(
        self,
        representations: list[torch.Tensor],
        independent_representations: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """``{"objective": objective(representations)}``.

        Args:
            representations: One encoded tensor per view.
            independent_representations: Unused.

        Returns:
            The loss dictionary.
        """
        return {"objective": self.objective(representations)}
