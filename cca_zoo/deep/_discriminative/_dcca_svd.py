from cca_zoo.deep._discriminative._dcca_ey import DCCA_EY
from cca_zoo.deep.objectives import _CCA_SVDLoss


class DCCA_SVD(DCCA_EY):
    """

    References
    ----------
    Chapman, James, Ana Lawry Aguila, and Lennie Wells. "A GeneralizedDeflation EigenGame with Extensions to Multiview Representation Learning." arXiv preprint arXiv:2211.11323 (2022).
    """

    def __init__(self, latent_dimensions: int, encoders=None, eps: float = 0, **kwargs):
        super().__init__(
            latent_dimensions=latent_dimensions, encoders=encoders, eps=eps, **kwargs
        )
        self.objective = _CCA_SVDLoss(eps=eps)
