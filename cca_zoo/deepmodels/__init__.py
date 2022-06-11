import cca_zoo.deepmodels.architectures
import cca_zoo.deepmodels.objectives
from ._dcca import DCCA
from ._dcca_barlow_twins import BarlowTwins
from ._dcca_noi import DCCA_NOI
from ._dcca_sdl import DCCA_SDL
from ._dccae import DCCAE
from ._dtcca import DTCCA
from ._dvcca import DVCCA
from ._splitae import SplitAE
from .utils import get_dataloaders

__all__ = [
    "DCCA",
    "DCCAE",
    "DCCA_NOI",
    "DCCA_SDL",
    "DVCCA",
    "BarlowTwins",
    "DTCCA",
    "SplitAE",
    "get_dataloaders",
]

classes = [
    "DCCA",
    "DCCAE",
    "DCCA_NOI",
    "DCCA_SDL",
    "DVCCA",
    "BarlowTwins",
    "DTCCA",
    "SplitAE",
]
