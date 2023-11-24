"""
Probabilistic CCA methods
"""
from ._cca import ProbabilisticCCA
from ._pls import ProbabilisticPLS
from ._plsregression import ProbabilisticPLSRegression
from ._rcca import ProbabilisticRCCA

__all__ = [
    "ProbabilisticCCA",
    "ProbabilisticPLSRegression",
    "ProbabilisticRCCA",
    "ProbabilisticPLS",
]
