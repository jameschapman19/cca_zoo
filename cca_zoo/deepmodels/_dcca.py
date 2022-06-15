import itertools
from typing import Iterable

import numpy as np
import torch

from cca_zoo.deepmodels import objectives
from cca_zoo.models import MCCA
from ._base import _BaseDeep
from ._callbacks import CorrelationCallback
from ..models._base import _BaseCCA


class DCCA(_BaseDeep, _BaseCCA):
    """
    A class used to fit a DCCA model.

    :Citation:

    Andrew, Galen, et al. "Deep canonical correlation analysis." International conference on machine learning. PMLR, 2013.

    """

    def __init__(
        self,
        latent_dims: int,
        objective=objectives.MCCA,
        encoders=None,
        r: float = 0,
        eps: float = 1e-5,
        **kwargs,
    ):
        """
        Constructor class for DCCA

        :param latent_dims: # latent dimensions
        :param objective: # CCA objective: normal tracenorm CCA by default
        :param encoders: list of encoder networks
        :param r: regularisation parameter of tracenorm CCA like ridge CCA. Needs to be VERY SMALL. If you get errors make this smaller
        :param eps: epsilon used throughout. Needs to be VERY SMALL. If you get errors make this smaller
        """
        super().__init__(latent_dims=latent_dims, **kwargs)
        self.encoders = torch.nn.ModuleList(encoders)
        self.objective = objective(latent_dims, r=r, eps=eps)

    def forward(self, views, **kwargs):
        """
        Forward method for the model. Outputs latent encoding for each view

        :param views:
        :param kwargs:
        :return:
        """
        z = []
        for i, encoder in enumerate(self.encoders):
            z.append(encoder(views[i]))
        return z

    def loss(self, views, **kwargs):
        """
        Define the loss function for the model. This is used by the DeepWrapper class

        :param views:
        :return:
        """
        z = self(views)
        return {"objective": self.objective.loss(z)}

    def post_transform(self, z, train=False):
        """
        Some models require a final linear CCA after model training.
        :param z: a list of all of the latent space embeddings for each view
        :param train: if the train flag is True this fits a new post transformation
        """
        if train:
            self.cca = MCCA(latent_dims=self.latent_dims)
            z = self.cca.fit_transform(z)
        else:
            z = self.cca.transform(z)
        return z

    def pairwise_correlations(
        self,
        loader: torch.utils.data.DataLoader,
        train=False,
    ):
        """
        Calculates correlation for entire batch from dataloader

        :param loader: a dataloader that matches the structure of that used for training
        :param train: whether to fit final linear transformation
        :return: by default returns the average pairwise correlation in each dimension (for 2 views just the correlation)
        """
        return _BaseCCA.pairwise_correlations(self, loader, train=train)

    def score(self, loader: torch.utils.data.DataLoader, train=False):
        """
        Returns average correlation in each dimension (averages over all pairs for multiview)

        :param loader: a dataloader that matches the structure of that used for training
        :param train: whether to fit final linear transformation
        """
        return _BaseCCA.score(self, loader, train=train)

    def configure_callbacks(self):
        return [CorrelationCallback()]
