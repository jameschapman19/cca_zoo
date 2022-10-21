from cca_zoo.deepmodels._discriminative import DCCA, DCCA_NOI, BarlowTwins, DCCA_SDL, DTCCA
from cca_zoo.deepmodels._generative import DVCCA, SplitAE, DCCAE
from cca_zoo.deepmodels.utils import architectures, objectives, callbacks
from cca_zoo.deepmodels.utils.utils import get_dataloaders

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
    "architectures",
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
