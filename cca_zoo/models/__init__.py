from ._grcca import GRCCA
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
from ._multiview import MCCA, KCCA, GCCA, KGCCA, TCCA, KTCCA
from ._ncca import NCCA
from ._partialcca import PartialCCA
from ._prcca import PRCCA
from ._rcca import rCCA, CCA, PLS

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
    "PRCCA",
    "GRCCA",
]

try:
    from ._stochastic import (
        PLSStochasticPower,
        IncrementalPLS,
        PLSGHAGEP,
        CCAGHAGEP,
        RCCAGHAGEP,
        PLSEigenGame,
        CCAEigenGame,
        RCCAEigenGame,
    )

    __all__.extend(
        "StochasticPowerPLS",
        "IncrementalPLS",
        "PLSGHAGEP",
        "CCAGHAGEP",
        "RCCAGHAGEP",
        "PLSEigenGame",
        "CCAEigenGame",
        "RCCAEigenGame",
    )
except:
    pass

classes = __all__
