"""Reusable sklearn ``_parameter_constraints`` fragments shared across models.

Kept separate from each model so the same documented range (e.g. a ridge
parameter in ``[0, 1]``) isn't repeated, and risked drifting, across every
class that has one.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any

from sklearn.utils._param_validation import Interval

#: A ridge/shrinkage parameter documented as a scalar or per-view list in
#: ``[0, 1]``. List elements are not individually checked, matching sklearn's
#: own shallow validation of list-typed parameters (e.g. ``sample_weight``).
RIDGE_PARAMETER: list[Any] = [Interval(Real, 0, 1, closed="both"), "array-like"]

#: A small positive constant added for numerical stability (e.g. to a
#: covariance matrix's eigenvalues before inversion).
POSITIVE_EPS: list[Any] = [Interval(Real, 0, None, closed="neither")]

#: A strictly positive iteration count.
POSITIVE_INT: list[Any] = [Interval(Integral, 1, None, closed="left")]
