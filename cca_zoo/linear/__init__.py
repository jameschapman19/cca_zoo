from ._gcca import GCCA
from ._gradient import CCAEY, CCAGH, CCASVD, PLSEY, PLSSVD, PLSStochasticPower
from ._grcca import GRCCA
from ._iterative import (
    PLS_ALS,
    SCCA_IPLS,
    SCCA_PMD,
    # AltMaxVar,
    ElasticCCA,
    SCCA_Parkhomenko,
    SCCA_Span,
)
from ._mcca import CCA, MCCA, rCCA
from ._partialcca import PartialCCA
from ._pcacca import PCACCA
from ._pls import MPLS, PLS
from ._prcca import PRCCA
from ._tcca import TCCA

__all__ = [
    "MCCA",
    "CCA",
    "rCCA",
    "PLS",
    "MPLS",
    "GCCA",
    "GRCCA",
    "PartialCCA",
    "PRCCA",
    "TCCA",
    "PCACCA",
    # "AltMaxVar",
    "SCCA_IPLS",
    "ElasticCCA",
    "PLS_ALS",
    "SCCA_PMD",
    "SCCA_Parkhomenko",
    "SCCA_Span",
    "CCAEY",
    "PLSEY",
    "CCAGH",
    "CCASVD",
    "PLSSVD",
    "PLSStochasticPower",
]
