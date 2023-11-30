from ._dcca import DCCA
from ..objectives import _CCA_EYLoss


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
        self.objective = _CCA_EYLoss(eps=eps)

    def loss(self, batch, **kwargs):
        # Encoding the representations with the forward method
        representations = self(batch["views"])
        if batch.get("independent_views") is None:
            independent_representations = None
        else:
            independent_representations = self(batch["independent_views"])
        return self.objective(representations, independent_representations)
