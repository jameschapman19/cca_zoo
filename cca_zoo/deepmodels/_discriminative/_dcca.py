import torch

from .. import objectives
from .._base import _BaseDeep
from ..callbacks import CorrelationCallback
from ...models import MCCA
from ...models._base import _BaseCCA


class DCCA(_BaseDeep, _BaseCCA):
    """
    A class used to fit a DCCA model.

    References
    ----------
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
        super().__init__(latent_dims=latent_dims, **kwargs)
        self.encoders = torch.nn.ModuleList(encoders)
        self.objective = objective(latent_dims, r=r, eps=eps)

    def forward(self, views, **kwargs):
        z = []
        for i, encoder in enumerate(self.encoders):
            z.append(encoder(views[i]))
        return z

    def loss(self, views, **kwargs):
        z = self(views)
        return {"objective": self.objective.loss(z)}

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

    def score(self, loader: torch.utils.data.DataLoader, **kwargs):
        """
        Returns average correlation in each dimension (averages over all pairs for multiview)

        :param **kwargs:
        :type **kwargs:
        :param loader: a dataloader that matches the structure of that used for training
        :param train: whether to fit final linear transformation
        """
        z = self.transform(loader)
        return MCCA(self.latent_dims).fit(z).score(z)

    def configure_callbacks(self):
        return [CorrelationCallback()]
