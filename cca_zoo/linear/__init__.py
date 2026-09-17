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
from ._moved import SAR as SAR
from ._moved import SCCAADMM as SCCAADMM
from ._moved import SCCAIPLS as SCCAIPLS
from ._moved import SCCAPMD as SCCAPMD
from ._moved import ParkhomenkoCCA as ParkhomenkoCCA
from ._moved import SCCASpan as SCCASpan
from ._moved import StochasticCCAEY as StochasticCCAEY
from ._moved import WaijenborgCCA as WaijenborgCCA
from ._partialcca import PartialCCA
from ._pls import PLS
from ._ransac_cca import RANSACCCA
from ._rcca import rCCA
from ._tcca import TCCA
from ._trimmed_cca import TrimmedCCA
from .gradient import CCA_EY as CCA_EY
from .gradient import CCAEY, PLSEY, HuberCCA
from .gradient import MCCA_EY as MCCA_EY
from .gradient import MCCAEY as MCCAEY
from .gradient import PLS_EY as PLS_EY

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
    # EY-loss (high-dimensional data)
    "PLSEY",
    "CCAEY",
    "HuberCCA",
]
# Deprecated aliases: CCA_EY, MCCA_EY, PLS_EY (underscored renames) and
# MCCAEY (CCAEY now supports 2 or more views directly, so the separate
# multiview class is gone) stay importable for backward compatibility via
# the `from .gradient import ...` statements above. SCCAPMD, SCCAADMM,
# SCCAIPLS, SCCASpan, WaijenborgCCA, ParkhomenkoCCA, SAR (moved to
# cca_zoo.sparse) and StochasticCCAEY (moved to cca_zoo.stochastic) stay
# importable here via `._moved` for the same reason. All are intentionally
# left out of __all__ (and therefore out of the API docs) since they are
# being removed from this module in a future release. PLSALS had no
# comparable use (an unregularised ALS baseline with no sparsity, no ridge,
# and no closed-form eigendecomposition counterpart already covering the
# same ground) and is dropped outright, with no deprecated alias.
