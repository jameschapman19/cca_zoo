from ._dcca import DCCA
from ..objectives import _GCCALoss


class DGCCA(DCCA):
    """
    A class used to fit a DGCCA model.

    Is just a thin wrapper round DCCA with the DGCCA objective

    References
    ----------


    """

    objective = _GCCALoss()

    def __init__(
        self, latent_dimensions: int, encoders=None, eps: float = 1e-5, **kwargs
    ):
        # Call the parent class constructor with the DGCCA objective function
        super().__init__(
            latent_dimensions=latent_dimensions, encoders=encoders, eps=eps, **kwargs
        )
