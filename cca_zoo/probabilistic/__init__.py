"""
Probabilistic CCA methods
"""
from ._cca import ProbabilisticCCA
from ._plsregression import ProbabilisticPLSRegression
from ._rcca import ProbabilisticRCCA
from ._pls import ProbabilisticPLS

__all__ = [
    "ProbabilisticCCA",
    "ProbabilisticPLSRegression",
    "ProbabilisticRCCA",
    "ProbabilisticPLS",
]
