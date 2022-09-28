from cca_zoo.deepmodels._discriminative import DCCA, DCCA_NOI, BarlowTwins, DCCA_SDL, DTCCA
from cca_zoo.deepmodels._generative import DVCCA, SplitAE, DCCAE
from cca_zoo.deepmodels._utils.utils import get_dataloaders
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
