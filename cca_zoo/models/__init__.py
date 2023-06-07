from ._rcca import CCA, PLS, rCCA
from ._mcca import KCCA, MCCA
from ._gcca import GCCA, KGCCA
from ._grcca import GRCCA
from ._ncca import NCCA
from ._partialcca import PartialCCA
from ._prcca import PRCCA
from ._tcca import KTCCA, TCCA

__all__ = [
    "CCA",
    "GCCA",
    "GRCCA",
    "KCCA",
    "MCCA",
    "NCCA",
    "PLS",
    "PRCCA",
    "KTCCA",
    "TCCA",
    "PartialCCA",
    "rCCA",
]

# if pytorch-lightning is installed then import ._iterative

try:
    from ._iterative import (
        CCAEY,
        CCAGH,
        CCASVD,
        PLS_ALS,  # SCCA_ADMM,; SWCCA,
        PLSEY,
        PLSSVD,
        SCCA_IPLS,
        SCCA_PMD,
        AltMaxVar,
        ElasticCCA,
        PLSStochasticPower,
        SCCA_Parkhomenko,
        SCCA_Span,
    )

    __all__ += [
        "ElasticCCA",
        "SCCA_IPLS",
        "PLS_ALS",
        "SCCA_PMD",
        "SCCA_Parkhomenko",
        "SCCA_Span",
        "AltMaxVar",
        "CCAEY",
        "PLSEY",
        "CCAGH",
        "CCASVD",
        "PLSSVD",
    ]
except:
    pass

classes = __all__
