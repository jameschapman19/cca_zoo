from ._mcca import MCCA, CCA, rCCA
from ._pls import PLS, MPLS
from ._gcca import GCCA
from ._grcca import GRCCA
from ._partialcca import PartialCCA
from ._prcca import PRCCA
from ._tcca import TCCA
from ._pcacca import PCACCA


__all__ = [
    "CCA",
    "GCCA",
    "GRCCA",
    "MCCA",
    "PLS",
    "PRCCA",
    "TCCA",
    "PartialCCA",
    "rCCA",
    "PCACCA",
    "MPLS",
]

# if pytorch-lightning is installed then import ._iterative

try:
    from ._iterative import (
        PLS_ALS,
        SCCA_IPLS,
        SCCA_PMD,
        AltMaxVar,
        ElasticCCA,
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
    ]
except:
    # let user know that if they want to use iterative methods they need to install pytorch-lightning from version 2.0.0
    print("To use iterative methods please install pytorch-lightning")
try:
    from ._gradient import (
        CCAEY,
        CCAGH,
        CCASVD,
        PLSEY,
        PLSSVD,
        PLSStochasticPower,
    )

    __all__ += [
        "CCAEY",
        "PLSEY",
        "CCAGH",
        "CCASVD",
        "PLSSVD",
        "PLSStochasticPower",
    ]

except:
    print("To use gradient methods please install pytorch-lightning")


classes = __all__
