from ._dcca import DCCA
from .. import objectives


class DTCCA(DCCA):
    """
    A class used to fit a DTCCA model.

    Is just a thin wrapper round DCCA with the DTCCA objective

    References
    ----------
    Wong, Hok Shing, et al. "Deep Tensor CCALoss for Multi-view Learning." IEEE Transactions on Big Data (2021).

    """

    def __init__(
        self,
        latent_dimensions: int,
        encoders=None,
        eps: float = 1e-5,
        **kwargs
    ):
        # Call the parent class constructor with the DTCCA objective function
        super().__init__(
            latent_dimensions=latent_dimensions,
            objective=objectives.TCCALoss,
            encoders=encoders,
            eps=eps,
            **kwargs
        )
