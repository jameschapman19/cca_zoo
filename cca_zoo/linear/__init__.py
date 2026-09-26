"""Linear CCA methods.

This module provides classical linear multiview CCA algorithms ranging from
the standard two-view CCA and PLS to multiset and generalised variants, as
well as EY-loss methods suited to high-dimensional data. Sparse/regularised
iterative methods live in :mod:`cca_zoo.sparse`; mini-batch methods live in
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
from ._rcca import rCCA
from ._tcca import TCCA
from ._trimmed_cca import TrimmedCCA
from .gradient import CCAEY, PLSEY, HuberCCA

__all__ = [
    # Exact eigendecomposition
    "CCA",
    "rCCA",
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
