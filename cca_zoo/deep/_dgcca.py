"""Deep generalized CCA."""

from __future__ import annotations

import torch.nn as nn

from cca_zoo.deep._dcca import DCCA
from cca_zoo.deep.objectives import GCCALoss


class DGCCA(DCCA):
    r"""Deep generalized CCA: correlate every view with a shared target.

    $$
    \mathcal{L} = -\sum_{d=1}^{k} \lambda_d\Bigl(\sum_i H_i H_i^\top\Bigr)
    $$

    for ridge-whitened encodings $H_i$
    (:class:`~cca_zoo.deep.objectives.GCCALoss`), for any number of views.

    Args:
        n_components: Latent dimension.
        encoders: One module per view.
        learning_rate: Adam learning rate. Default is 1e-3.
        eps: Whitening ridge. Default is 1e-6.

    References:
        Benton, A., Khayrallah, H., Gujral, B., Reisinger, D. A., Zhang, S.,
        & Arora, R. (2019). Deep generalized canonical correlation analysis.
        RepL4NLP.

    Examples:
        >>> import torch.nn as nn
        >>> from cca_zoo.deep import DGCCA
        >>> encoders = [nn.Linear(10, 4), nn.Linear(8, 4), nn.Linear(6, 4)]
        >>> model = DGCCA(n_components=4, encoders=encoders)
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
            objective=GCCALoss(eps=eps),
            learning_rate=learning_rate,
            eps=eps,
        )
