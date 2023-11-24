"""
Deep CCA methods
"""
from . import architectures
from . import architectures, objectives
from . import data
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
    "DCCA_GHA",
    "DCCA_SVD",
    "DGCCA",
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
]
