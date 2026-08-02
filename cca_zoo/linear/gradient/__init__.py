"""Gradient-descent CCA variants for high-dimensional and streaming data.

These methods optimise the unconstrained Eckart-Young (EY) objective by
mini-batch momentum gradient descent, replacing the full covariance-matrix
eigendecomposition used by the exact linear models. See
:mod:`cca_zoo._utils._ey` for the shared EY-loss machinery.

Classes:
    PLS_EY: Eckart-Young PLS.
    CCA_EY: Eckart-Young CCA (whitened).
    MCCA_EY: Multiview extension of CCA_EY (>=2 views).
"""

from cca_zoo.linear.gradient._cca_ey import CCA_EY
from cca_zoo.linear.gradient._mcca_ey import MCCA_EY
from cca_zoo.linear.gradient._pls_ey import PLS_EY

__all__ = ["PLS_EY", "CCA_EY", "MCCA_EY"]
