from ._dcca import DCCA
from ..objectives import _TCCALoss
from ...linear._tcca import TCCA


class DTCCA(TCCA, DCCA):
    """
    A class used to fit a DTCCA model.

    Is just a thin wrapper round DCCA with the DTCCA objective

    References
    ----------
    Wong, Hok Shing, et al. "Deep Tensor CCA for Multi-view Learning." IEEE Transactions on Big Data (2021).

    """

    objective = _TCCALoss()

    def __init__(
        self, latent_dimensions: int, encoders=None, eps: float = 1e-5, **kwargs
    ):
        # Initialize DCCA part with DTCCA objective function
        DCCA.__init__(
            self,
            latent_dimensions=latent_dimensions,
            encoders=encoders,
            eps=eps,
            **kwargs
        )
