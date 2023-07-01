from ._mcca import MCCA, CCA, rCCA
from ._pls import PLS, MPLS
from ._gcca import GCCA
from ._grcca import GRCCA
from ._partialcca import PartialCCA
from ._prcca import PRCCA
from ._tcca import TCCA
from ._pcacca import PCACCA
from ._iterative import (
    AltMaxVar,
    SCCA_IPLS,
    ElasticCCA,
    PLS_ALS,
    SCCA_PMD,
    SCCA_Parkhomenko,
    SCCA_Span,
)
from ._gradient import CCAEY, PLSEY, CCAGH, CCASVD, PLSSVD, PLSStochasticPower

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
    "AltMaxVar",
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
