from . import architectures, callbacks, objectives
from ._discriminative import (
    DCCA,
    DCCA_EY,
    DCCA_GHA,
    DCCA_NOI,
    DCCA_SDL,
    DCCA_SVD,
    DGCCA,
    DTCCA,
    BarlowTwins,
)
from ._generative import DCCAE, DVCCA, SplitAE

__all__ = [
    "DCCA",
    "DCCAE",
    "DCCA_NOI",
    "DCCA_SDL",
    "DVCCA",
    "BarlowTwins",
    "DTCCA",
    "DCCA_EY",
    "SplitAE",
    "architectures",
    "objectives",
    "callbacks",
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
    "DCCA_EY",
]
