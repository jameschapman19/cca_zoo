import torch

from cca_zoo.deep import objectives
from cca_zoo.deep._base import BaseDeep
from cca_zoo.linear._mcca import MCCA


class DCCA(BaseDeep):
    """
    A class used to fit a DCCA model.

    References
    ----------
    Andrew, Galen, et al. "Deep canonical correlation analysis." International conference on machine learning. PMLR, 2013.

    """

    def __init__(
        self,
        latent_dimensions: int,
        objective=objectives.MCCA,
        encoders=None,
        r: float = 0,
        eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__(latent_dimensions=latent_dimensions, **kwargs)
        # Check if encoders are provided and have the same length as the number of views
        if encoders is None:
            raise ValueError(
                "Encoders must be a list of torch.nn.Module with length equal to the number of views."
            )
        self.encoders = torch.nn.ModuleList(encoders)
        self.objective = objective(r=r, eps=eps)

    def forward(self, views, **kwargs):
        if not hasattr(self, "n_views_"):
            self.n_views_ = len(views)
        # Use list comprehension to encode each view
        z = [encoder(view) for encoder, view in zip(self.encoders, views)]
        return z

    def loss(self, views, **kwargs):
        z = self(views)
        return {"objective": self.objective.loss(z)}

    def pairwise_correlations(self, loader: torch.utils.data.DataLoader):
        # Call the parent class method
        return super().pairwise_correlations(loader)

    def correlation_captured(self, z):
        return MCCA(latent_dimensions=self.latent_dimensions).fit(z).score(z).sum()

    def score(self, loader: torch.utils.data.DataLoader, **kwargs):
        z = self.transform(loader)
        corr = self.correlation_captured(z)
        return corr
