from ._altmaxvar import AltMaxVar
from ._elastic import ElasticCCA, SCCA_IPLS
from ._pddgcca import PDD_GCCA
from ._pls_als import PLS_ALS
from ._pmd import SCCA_PMD
from ._scca_admm import SCCA_ADMM
from ._scca_parkhomenko import SCCA_Parkhomenko
from ._spancca import SCCA_Span
from ._swcca import SWCCA

__all__ = [
    "AltMaxVar",
    "ElasticCCA",
    "SCCA_IPLS",
    "PDD_GCCA",
    "PLS_ALS",
    "SCCA_PMD",
    "SCCA_ADMM",
    "SCCA_Parkhomenko",
    "SCCA_Span",
    "SWCCA",
]
