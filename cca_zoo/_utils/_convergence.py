"""Convergence reporting for the iterative models, as in sklearn."""

from __future__ import annotations

import warnings

from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning


def warn_if_not_converged(estimator: BaseEstimator, converged: bool) -> None:
    """Warn that ``estimator`` stopped at ``max_iter`` before meeting its tolerance."""
    if not converged:
        warnings.warn(
            f"{type(estimator).__name__} did not converge in "
            f"max_iter={estimator.get_params()['max_iter']} iterations. Increase "
            "max_iter, or scale the views with StandardScaler.",
            ConvergenceWarning,
            stacklevel=3,
        )
