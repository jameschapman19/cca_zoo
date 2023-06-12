from ._dcca import DCCA
from .. import objectives


class DGCCA(DCCA):
    """
    A class used to fit a DGCCA model.

    Is just a thin wrapper round DCCA with the DGCCA objective

    References
    ----------


    """

    def __init__(
        self,
        latent_dimensions: int,
        encoders=None,
        r: float = 0,
        eps: float = 1e-5,
        **kwargs
    ):
        # Call the parent class constructor with the DGCCA objective function
        super().__init__(
            latent_dimensions=latent_dimensions,
            objective=objectives.GCCA,
            encoders=encoders,
            r=r,
            eps=eps,
            **kwargs
        )
