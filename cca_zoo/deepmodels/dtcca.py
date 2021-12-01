from typing import Iterable

import numpy as np

from cca_zoo.deepmodels import objectives
from cca_zoo.deepmodels.dcca import DCCA
from cca_zoo.models import TCCA


class DTCCA(DCCA):
    """
    A class used to fit a DTCCA model.

    Is just a thin wrapper round DCCA with the DTCCA objective and a TCCA post-processing

    :Citation:

    Wong, Hok Shing, et al. "Deep Tensor CCA for Multi-view Learning." IEEE Transactions on Big Data (2021).

    """

    def __init__(
        self, latent_dims: int, encoders=None, r: float = 0, eps: float = 1e-3
    ):
        """

        :param latent_dims: # latent dimensions
        :param encoders: list of encoder networks
        :param r: regularisation parameter of tracenorm CCA like ridge CCA. Needs to be VERY SMALL. If you get errors make this smaller
        :param eps: epsilon used throughout. Needs to be VERY SMALL. If you get errors make this smaller
        """
        super().__init__(
            latent_dims=latent_dims,
            objective=objectives.TCCA,
            encoders=encoders,
            r=r,
            eps=eps,
        )

    def post_transform(self, z_list, train=False) -> Iterable[np.ndarray]:
        if train:
            self.cca = TCCA(latent_dims=self.latent_dims)
            z_list = self.cca.fit_transform(z_list)
        else:
            z_list = self.cca.transform(z_list)
        return z_list
