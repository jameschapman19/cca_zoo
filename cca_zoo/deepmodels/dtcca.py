from typing import Iterable

import torch

from cca_zoo.deepmodels import objectives
from cca_zoo.deepmodels.architectures import BaseEncoder, Encoder
from cca_zoo.deepmodels.dcca import DCCA
from cca_zoo.models import TCCA


class DTCCA(DCCA, torch.nn.Module):
    """
    A class used to fit a DTCCA model.

    Examples
    --------
    >>> from cca_zoo.deepmodels import DTCCA
    >>> model = DTCCA()
    """

    def __init__(self, latent_dims: int, encoders: Iterable[BaseEncoder] = [Encoder, Encoder],
                 learning_rate=1e-3, r: float = 0,
                 schedulers: Iterable = None, optimizers: Iterable = None):
        """

        :param latent_dims:
        :param encoders:
        :param learning_rate:
        :param r:
        :param schedulers:
        :param optimizers:
        """
        super().__init__(latent_dims, objective=objectives.TCCA, encoders=encoders, learning_rate=learning_rate, r=r,
                         schedulers=schedulers, optimizers=optimizers)

    def post_transform(self, *z_list, train=False):
        if train:
            self.cca = TCCA(latent_dims=self.latent_dims)
            self.cca.fit(*z_list)
            z_list = self.cca.transform(*z_list)
        else:
            z_list = self.cca.transform(*z_list)
        return z_list
