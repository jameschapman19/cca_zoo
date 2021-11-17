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
        We use the forward model to define the transformation of views to the latent space

        :param args: batches for each view separated by commas
        """
        raise NotImplementedError

    @abstractmethod
    def loss(self, *args, **kwargs):
        """
        Required when using the LightningTrainer
        """
        raise NotImplementedError

    def post_transform(self, z_list, train=False) -> Iterable[np.ndarray]:
        """
        Some models require a final linear CCA after model training.

        :param z_list: a list of all of the latent space embeddings for each view
        :param train: if the train flag is True this fits a new post transformation
        """
        return z_list
