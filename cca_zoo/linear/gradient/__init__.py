"""EY-loss CCA variants for high-dimensional data.

These methods optimise the unconstrained Eckart-Young (EY) objective,
replacing the full covariance-matrix eigendecomposition used by the exact
linear models. See :mod:`cca_zoo._utils._ey` for the shared EY-loss
machinery.

Classes:
    PLSEY: Eckart-Young PLS, full-batch.
    CCAEY: Eckart-Young CCA (2 or more views), full-batch.
    HuberCCA: Bounded-influence (Huber-style) extension of CCAEY, full-batch.

Mini-batch fitting (:class:`~cca_zoo.stochastic.StochasticCCAEY`) lives in
:mod:`cca_zoo.stochastic`.
"""

from cca_zoo.linear.gradient._cca_ey import CCAEY
from cca_zoo.linear.gradient._huber_cca import HuberCCA
from cca_zoo.linear.gradient._pls_ey import PLSEY

__all__ = ["PLSEY", "CCAEY", "HuberCCA"]
