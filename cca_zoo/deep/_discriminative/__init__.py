from ._dcca import DCCA
from ._dcca_barlow_twins import BarlowTwins
from ._dcca_ey import DCCA_EY
from ._dcca_gha import DCCA_GHA
from ._dcca_noi import DCCA_NOI
from ._dcca_sdl import DCCA_SDL
from ._dcca_svd import DCCA_SVD
from ._dgcca import DGCCA
from ._dtcca import DTCCA

__all__ = [
    "DCCA",
    "DCCA_EY",
    "DCCA_GHA",
    "DCCA_SVD",
    "DCCA_NOI",
    "DCCA_SDL",
    "DTCCA",
    "BarlowTwins",
    "DGCCA",
]
