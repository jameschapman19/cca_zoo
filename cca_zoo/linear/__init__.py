"""Linear CCA.

Closed-form two-view and multiview models, reduced-rank-regression and
robust variants, and Eckart-Young models fitted by L-BFGS-B. Sparse
models are in :mod:`cca_zoo.sparse`, mini-batch ones in
:mod:`cca_zoo.stochastic`.
"""

from ._cca import CCA
from ._ccar3 import CCAR3
from ._ecca import ECCA
from ._gcca import GCCA
from ._graphical_lasso_cca import GraphicalLassoCCA
from ._grcca import GRCCA
from ._mcca import MCCA
from ._partialcca import PartialCCA
from ._pls import PLS
from ._projection_pursuit_cca import ProjectionPursuitCCA
from ._ransac_cca import RANSACCCA
from ._ridge_cca import RidgeCCA
from ._tcca import TCCA
from ._trimmed_cca import TrimmedCCA
from .gradient import CCAEY, PLSEY, HuberCCA

__all__ = [
    # Exact eigendecomposition
    "CCA",
    "RidgeCCA",
    "PLS",
    "MCCA",
    "GCCA",
    "TCCA",
    # Confound-adjusted / structured
    "PartialCCA",
    "GRCCA",
    # Reduced-rank regression
    "CCAR3",
    "ECCA",
    # Sparse-precision within-view covariance
    "GraphicalLassoCCA",
    # Robust to contaminated training data
    "RANSACCCA",
    "TrimmedCCA",
    "ProjectionPursuitCCA",
    # EY-loss (high-dimensional data)
    "PLSEY",
    "CCAEY",
    "HuberCCA",
]
