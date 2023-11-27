"""
The :mod:`cca_zoo.linear` module includes a variety of linear CCA algorithms.
"""
from ._gcca import GCCA
from ._grcca import GRCCA
from ._mcca import CCA, MCCA, rCCA
from ._partialcca import PartialCCA
from ._pcacca import PCACCA
from ._pls import MPLS, PLS
from ._prcca import PRCCA
from ._tcca import TCCA
from ._gradient._ey import CCA_EY, PLS_EY
from ._gradient._gha import CCA_GHA
from ._gradient._svd import CCA_SVD
from ._gradient._stochasticpls import PLSStochasticPower
from ._iterative._elastic import SCCA_IPLS, ElasticCCA
from ._iterative._pls_als import PLS_ALS
from ._iterative._scca_parkhomenko import SCCA_Parkhomenko
from ._iterative._scca_span import SCCA_Span
from ._iterative._spls import SPLS

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
    "SCCA_IPLS",
    "ElasticCCA",
    "PLS_ALS",
    "SPLS",
    "SCCA_Parkhomenko",
    "SCCA_Span",
    "CCA_EY",
    "PLS_EY",
    "CCA_GHA",
    "CCA_SVD",
    "PLSStochasticPower",
]
