from ._elastic import ElasticCCA, SCCA_IPLS
from ._gradkcca import GradKCCA
from ._pdd import AltMaxVar
from ._pls_als import PLS_ALS
from ._pmd import SCCA_PMD
from ._scca_admm import SCCA_ADMM
from ._scca_hsic import SCCA_HSIC
from ._scca_parkhomenko import SCCA_Parkhomenko
from ._spancca import SCCA_Span
from ._swcca import SWCCA

__all__ = [
    "ElasticCCA",
    "SCCA_IPLS",
    "PLS_ALS",
    "SCCA_PMD",
    "SCCA_ADMM",
    "SCCA_Parkhomenko",
    "SCCA_Span",
    "SWCCA",
    "GradKCCA",
    "SCCA_HSIC",
    "AltMaxVar",
]
