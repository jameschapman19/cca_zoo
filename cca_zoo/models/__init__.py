from ._gcca import GCCA, KGCCA
from ._iterative import (
    PLS_ALS,
    SCCA_PMD,
    ElasticCCA,
    SCCA_Parkhomenko,
    SCCA_IPLS,
    SCCA_ADMM,
    SCCA_Span,
    SWCCA,
)
from ._mcca import MCCA, KCCA
from ._ncca import NCCA
from ._partialcca import PartialCCA
from ._rcca import rCCA, CCA, PLS
from ._tcca import TCCA, KTCCA

try:
    from ._stochastic import StochasticPowerPLS, IncrementalPLS
except:
    pass

__all__ = [
    "GCCA",
    "KGCCA",
    "PLS_ALS",
    "SCCA_PMD",
    "ElasticCCA",
    "SCCA_Parkhomenko",
    "SCCA_IPLS",
    "SCCA_ADMM",
    "SCCA_Span",
    "SWCCA",
    "MCCA",
    "KCCA",
    "NCCA",
    "PartialCCA",
    "rCCA",
    "CCA",
    "PLS",
    "TCCA",
    "KTCCA",
    "StochasticPowerPLS",
    "IncrementalPLS",
]

classes = __all__
