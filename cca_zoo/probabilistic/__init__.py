"""
Probabilistic CCA methods
"""
try:
    import numpyro
except ModuleNotFoundError:
    print(
        "Warning: numpyro is not installed. Some functionality in cca_zoo.probabilistic may be limited."
    )
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
