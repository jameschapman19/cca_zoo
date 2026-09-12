"""Gradient-descent CCA variants for high-dimensional and streaming data.

These methods optimise the unconstrained Eckart-Young (EY) objective by
mini-batch momentum gradient descent, replacing the full covariance-matrix
eigendecomposition used by the exact linear models. See
:mod:`cca_zoo._utils._ey` for the shared EY-loss machinery.

Classes:
    PLSEY: Eckart-Young PLS.
    CCAEY: Eckart-Young CCA (whitened).
    MCCAEY: Multiview extension of CCAEY (>=2 views).
"""

from cca_zoo.linear.gradient._cca_ey import CCA_EY as CCA_EY
from cca_zoo.linear.gradient._cca_ey import CCAEY
from cca_zoo.linear.gradient._mcca_ey import MCCA_EY as MCCA_EY
from cca_zoo.linear.gradient._mcca_ey import MCCAEY
from cca_zoo.linear.gradient._pls_ey import PLS_EY as PLS_EY
from cca_zoo.linear.gradient._pls_ey import PLSEY

__all__ = ["PLSEY", "CCAEY", "MCCAEY"]
# Deprecated aliases PLS_EY, CCA_EY, MCCA_EY stay importable for backward
# compatibility but are intentionally left out of __all__/docs.
