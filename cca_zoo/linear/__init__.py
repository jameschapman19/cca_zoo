"""Linear CCA methods.

This module provides classical linear multiview CCA algorithms ranging from
the standard two-view CCA and PLS to multiset and generalised variants, as
well as sparse/regularised iterative methods and EY-loss methods suited to
high-dimensional data.
"""

from ._cca import CCA
from ._ccar3 import CCAR3
from ._gcca import GCCA
from ._grcca import GRCCA
from ._iterative import PLS_ALS as PLS_ALS
from ._iterative import (
    PLSALS,
    SAR,
    SCCAADMM,
    SCCAIPLS,
    SCCAPMD,
    ElasticCCA,
    ParkhomenkoCCA,
    SCCASpan,
)
from ._iterative import SCCA_ADMM as SCCA_ADMM
from ._iterative import SCCA_IPLS as SCCA_IPLS
from ._iterative import SCCA_PMD as SCCA_PMD
from ._iterative import SCCA_Span as SCCA_Span
from ._mcca import MCCA
from ._partialcca import PartialCCA
from ._pls import PLS
from ._ransac_cca import RANSACCCA
from ._rcca import rCCA
from ._tcca import TCCA
from .gradient import CCA_EY as CCA_EY
from .gradient import CCAEY, PLSEY, HuberCCA, StochasticCCAEY
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
    # Robust to contaminated training data
    "RANSACCCA",
    # EY-loss (high-dimensional data)
    "PLSEY",
    "CCAEY",
    "StochasticCCAEY",
    "HuberCCA",
    # Sparse / regularised ALS
    "SCCAPMD",
    "SCCAADMM",
    "SCCAIPLS",
    "SCCASpan",
    "ElasticCCA",
    "ParkhomenkoCCA",
    "SAR",
    "PLSALS",
]
# Deprecated aliases (CCA_EY, MCCA_EY, PLS_EY, SCCA_PMD, SCCA_ADMM,
# SCCA_IPLS, SCCA_Span, PLS_ALS -- underscored renames -- and MCCAEY, whose
# functionality CCAEY now provides directly) stay importable for backward
# compatibility via the `from .gradient import ...` / `from ._iterative
# import ...` statements above, but are intentionally left out of __all__
# (and therefore out of the API docs) since they are being removed in a
# future release.
