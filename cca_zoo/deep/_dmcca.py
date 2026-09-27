"""Deep multiset CCA."""

from __future__ import annotations

import torch.nn as nn

from cca_zoo.deep._dcca import DCCA
from cca_zoo.deep.objectives import MCCALoss


class DMCCA(DCCA):
    r"""Deep multiset CCA: the sum of pairwise deep CCA losses.

    $$
    \mathcal{L} = \sum_{i < j} \mathcal{L}_{\text{CCA}}(z_i, z_j)
    $$

    (:class:`~cca_zoo.deep.objectives.MCCALoss`), for any number of views.

    Args:
        n_components: Latent dimension.
        encoders: One module per view.
        learning_rate: Adam learning rate. Default is 1e-3.
        eps: Ridge of each pairwise loss. Default is 1e-6.

    References:
        Somandepalli, K., Kumar, N., Travadi, R., & Narayanan, S. (2019).
        Multimodal representation learning using deep multiset canonical
        correlation analysis. arXiv:1904.01775.

    Examples:
        >>> import torch.nn as nn
        >>> from cca_zoo.deep import DMCCA
        >>> encoders = [nn.Linear(10, 4), nn.Linear(8, 4), nn.Linear(6, 4)]
        >>> model = DMCCA(n_components=4, encoders=encoders)
    """

    def __init__(
        self,
        n_components: int,
        encoders: list[nn.Module],
        learning_rate: float = 1e-3,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(
            n_components=n_components,
            encoders=encoders,
            objective=MCCALoss(eps=eps),
            learning_rate=learning_rate,
            eps=eps,
        )
