from ._dcca import DCCA
from ..objectives import CCA_EYLoss


class DCCA_EY(DCCA):
    """

    References
    ----------
    Chapman, James, Ana Lawry Aguila, and Lennie Wells. "A Generalized EigenGame with Extensions to Multiview Representation Learning." arXiv preprint arXiv:2211.11323 (2022).
    """

    def __init__(self, latent_dimensions: int, encoders=None, eps: float = 0, **kwargs):
        super().__init__(
            latent_dimensions=latent_dimensions, encoders=encoders, eps=eps, **kwargs
        )
        self.objective = CCA_EYLoss(eps=eps)

    def loss(self, batch, **kwargs):
        # Encoding the representations with the forward method
        z = self(batch["views"])
        independent_views = batch.get("independent_views", None)
        return self.objective.loss(z, independent_views)
