# from ._altmaxvar import AltMaxVar
from ._elastic import SCCA_IPLS, ElasticCCA
from ._pls_als import PLS_ALS
from ._scca_parkhomenko import SCCA_Parkhomenko
from ._scca_span import SCCA_Span
from ._spls import SPLS

__all__ = [
    "ElasticCCA",
    "SCCA_IPLS",
    "PLS_ALS",
    "SPLS",
    "SCCA_Parkhomenko",
    "SCCA_Span",
    # "AltMaxVar",
]
