from ._mcca import MCCA, CCA, PLS, rCCA
from ._gcca import GCCA
from ._grcca import GRCCA
from ._ncca import NCCA
from ._partialcca import PartialCCA
from ._prcca import PRCCA
from ._tcca import TCCA
from ._pcacca import PCACCA
from ._kcca import KCCA, KGCCA, KTCCA


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
    "PCACCA",
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
    # let user know that if they want to use iterative methods they need to install pytorch-lightning from version 2.0.0
    print(
        "To use iterative methods please install pytorch-lightning from version 2.0.0"
    )

classes = __all__
