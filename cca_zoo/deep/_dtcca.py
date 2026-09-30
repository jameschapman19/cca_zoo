"""Deep tensor CCA."""

from __future__ import annotations

import torch.nn as nn

from cca_zoo.deep._dcca import _ObjectiveModel
from cca_zoo.deep.objectives import TCCALoss


class DTCCA(_ObjectiveModel):
    r"""Deep tensor CCA: maximise the cross-moment tensor of whitened encodings.

    Minimises $-\|M\|_F$ with
    $M = \frac{1}{n} \sum_s H_1[s] \otimes \cdots \otimes H_M[s]$ for
    ridge-whitened encodings $H_i$
    (:class:`~cca_zoo.deep.objectives.TCCALoss`).

    Args:
        n_components: Number of latent dimensions.
        encoders: One module per view.
        learning_rate: Adam learning rate. Default is 1e-3.
        reg_covar: Non-negative regularisation added to the diagonal of each
            covariance, as scikit-learn's ``GaussianMixture``. Default is 1e-6.

    References:
        Wong, H. S., Wang, L., Chan, R., & Zeng, T. (2021). Deep tensor
        CCA for multi-view learning. IEEE Transactions on Big Data.

    Examples:
        >>> import torch.nn as nn
        >>> from cca_zoo.deep import DTCCA
        >>> encoders = [nn.Linear(10, 4), nn.Linear(8, 4), nn.Linear(6, 4)]
        >>> model = DTCCA(n_components=4, encoders=encoders)
    """

    def __init__(
        self,
        n_components: int,
        encoders: list[nn.Module],
        learning_rate: float = 1e-3,
        reg_covar: float = 1e-6,
    ) -> None:
        super().__init__(
            n_components=n_components,
            encoders=encoders,
            learning_rate=learning_rate,
        )
        self.reg_covar = reg_covar
        self.objective = TCCALoss(reg_covar=reg_covar)
