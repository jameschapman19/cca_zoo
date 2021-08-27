from abc import abstractmethod
from typing import Iterable

import numpy as np
from torch import nn


class _DCCA_base(nn.Module):
    def __init__(self, latent_dims: int):
        super().__init__()
        self.latent_dims = latent_dims

    @abstractmethod
    def forward(self, *args):
        """
        :param args: batches for each view separated by commas
        :return: views encoded to latent dimensions
        """
        raise NotImplementedError

    def post_transform(self, *z_list, train=False) -> Iterable[np.ndarray]:
        return z_list
