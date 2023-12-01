"""
The :mod:`cca_zoo.deep` module includes a variety of deep CCA algorithms.
"""
import importlib.util

if (
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("lightning") is None
):
    print(
        "Warning: torch or lightning are not installed. Some functionality in cca_zoo.deep may be limited."
    )
from . import architectures, objectives
from ._discriminative._dcca import DCCA
from ._discriminative._barlow_twins import BarlowTwins
from ._discriminative._dcca_ey import DCCA_EY
from ._discriminative._dcca_gha import DCCA_GHA
from ._discriminative._dcca_noi import DCCA_NOI
from ._discriminative._dcca_sdl import DCCA_SDL
from ._discriminative._dcca_svd import DCCA_SVD
from ._discriminative._dgcca import DGCCA
from ._discriminative._dtcca import DTCCA
from ._discriminative._vicreg import VICReg
from ._generative._dccae import DCCAE
from ._generative._dvcca import DVCCA
from ._generative._splitae import SplitAE

__all__ = [
    "DCCA",
    "DCCA_GHA",
    "DCCA_SVD",
    "DGCCA",
    "DCCAE",
    "DCCA_NOI",
    "DCCA_SDL",
    "DVCCA",
    "BarlowTwins",
    "VICReg",
    "DTCCA",
    "DCCA_EY",
    "SplitAE",
    "architectures",
    "objectives",
]
