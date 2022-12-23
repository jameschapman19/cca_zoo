from typing import Iterable

import numpy as np

from ._dcca import DCCA
from .. import objectives
from ...models import TCCA


class DTCCA(DCCA):
    """
    A class used to fit a DTCCA model.

    Is just a thin wrapper round DCCA with the DTCCA objective and a TCCA post-processing

    References
    ----------
    Wong, Hok Shing, et al. "Deep Tensor CCA for Multi-view Learning." IEEE Transactions on Big Data (2021).

    """

    def __init__(
        self, latent_dims: int, encoders=None, r: float = 0, eps: float = 1e-5, **kwargs
    ):
        super().__init__(
            latent_dims=latent_dims,
            objective=objectives.TCCA,
            encoders=encoders,
            r=r,
            eps=eps,
            **kwargs
        )

    def post_transform(self, z, train=False) -> Iterable[np.ndarray]:
        if train:
            self.cca = TCCA(latent_dims=self.latent_dims)
            z = self.cca.fit_transform(z)
        else:
            z = self.cca.transform(z)
        return z
