"""EY-loss CCA variants for high-dimensional data.

These methods optimise the unconstrained Eckart-Young (EY) objective,
replacing the full covariance-matrix eigendecomposition used by the exact
linear models. See :mod:`cca_zoo._utils._ey` for the shared EY-loss
machinery.

Classes:
    PLSEY: Eckart-Young PLS, full-batch.
    CCAEY: Eckart-Young CCA (2 or more views), full-batch.
    StochasticCCAEY: CCAEY fit by mini-batch momentum SGD instead.
    HuberCCA: Bounded-influence (Huber-style) extension of CCAEY, full-batch.
"""

from cca_zoo.linear.gradient._cca_ey import CCA_EY as CCA_EY
from cca_zoo.linear.gradient._cca_ey import CCAEY
from cca_zoo.linear.gradient._cca_ey import MCCA_EY as MCCA_EY
from cca_zoo.linear.gradient._cca_ey import MCCAEY as MCCAEY
from cca_zoo.linear.gradient._huber_cca import HuberCCA
from cca_zoo.linear.gradient._pls_ey import PLS_EY as PLS_EY
from cca_zoo.linear.gradient._pls_ey import PLSEY
from cca_zoo.linear.gradient._stochastic_cca_ey import StochasticCCAEY

__all__ = ["PLSEY", "CCAEY", "StochasticCCAEY", "HuberCCA"]
# Deprecated aliases PLS_EY, CCA_EY, MCCA_EY, and MCCAEY (CCAEY now supports
# 2 or more views directly, so the separate multiview class is gone) stay
# importable for backward compatibility but are intentionally left out of
# __all__/docs.
