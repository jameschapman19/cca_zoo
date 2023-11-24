from ._dcca import DCCA
from ..objectives import _MCCALoss


class DMCCA(DCCA):
    """
    A class used to fit a DMCCA model.

    Is just a thin wrapper round DCCA with the DMCCA objective

    References
    ----------


    """

    objective = _MCCALoss()

    def __init__(
        self, latent_dimensions: int, encoders=None, eps: float = 1e-5, **kwargs
    ):
        # Call the parent class constructor with the DMCCA objective function
        super().__init__(
            latent_dimensions=latent_dimensions, encoders=encoders, eps=eps, **kwargs
        )
