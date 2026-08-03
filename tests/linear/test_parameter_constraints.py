"""Tests for sklearn-style constructor parameter validation.

``BaseModel`` and several subclasses declare ``_parameter_constraints`` and
call ``self._validate_params()`` from ``_setup_fit()`` (see cca_zoo/_base.py).
Invalid constructor parameters should raise
``sklearn.utils._param_validation.InvalidParameterError`` (a ``ValueError``
subclass) as soon as ``fit`` is called.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.utils._param_validation import InvalidParameterError

from cca_zoo.linear import CCA, CCAR3, GCCA, GRCCA, MCCA, TCCA, PartialCCA, rCCA

# ---------------------------------------------------------------------------
# Base parameters (latent_dimensions, center), shared by every model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("latent_dimensions", [0, -1, 1.5])
def test_invalid_latent_dimensions_rejected(
    latent_dimensions: object, two_views: list[np.ndarray]
) -> None:
    """A non-positive or non-integer latent_dimensions raises."""
    with pytest.raises(InvalidParameterError):
        CCA(latent_dimensions=latent_dimensions).fit(two_views)  # type: ignore[arg-type]


def test_invalid_center_rejected(two_views: list[np.ndarray]) -> None:
    """A non-boolean center raises."""
    with pytest.raises(InvalidParameterError):
        CCA(center="yes").fit(two_views)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# c: ridge parameter in [0, 1] — rCCA, MCCA, GCCA, and (via inheritance)
# PartialCCA / GRCCA
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", [rCCA, MCCA, GCCA])
@pytest.mark.parametrize("c", [-0.1, 1.1])
def test_invalid_c_rejected(
    ModelClass: type, c: float, two_views: list[np.ndarray]
) -> None:
    """C outside [0, 1] raises for every model that documents that range."""
    with pytest.raises(InvalidParameterError):
        ModelClass(c=c).fit(two_views)


def test_partialcca_inherits_mcca_c_constraint(two_views: list[np.ndarray]) -> None:
    """PartialCCA doesn't redeclare constraints; it inherits MCCA's via the MRO."""
    rng = np.random.default_rng(1)
    partials = rng.standard_normal((50, 3))
    with pytest.raises(InvalidParameterError):
        PartialCCA(c=2.0).fit(two_views, partials=partials)


def test_grcca_inherits_mcca_c_constraint(two_views: list[np.ndarray]) -> None:
    """GRCCA doesn't redeclare constraints; it inherits MCCA's via the MRO."""
    with pytest.raises(InvalidParameterError):
        GRCCA(c=-1.0).fit(two_views)


# ---------------------------------------------------------------------------
# Model-specific extra parameters
# ---------------------------------------------------------------------------


def test_mcca_invalid_pca_rejected(two_views: list[np.ndarray]) -> None:
    """A non-boolean pca flag raises."""
    with pytest.raises(InvalidParameterError):
        MCCA(pca="yes").fit(two_views)  # type: ignore[arg-type]


def test_mcca_invalid_eps_rejected(two_views: list[np.ndarray]) -> None:
    """A non-positive eps raises."""
    with pytest.raises(InvalidParameterError):
        MCCA(eps=0.0).fit(two_views)


def test_tcca_invalid_random_state_rejected(three_views: list[np.ndarray]) -> None:
    """A negative or non-integer random_state raises."""
    with pytest.raises(InvalidParameterError):
        TCCA(random_state=-1).fit(three_views)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lambda_": -1.0},
        {"highdim": "nope"},
        {"ledoit_wolf": "nope"},
        {"rho": 0.0},
        {"max_iter": 0},
        {"tol": 0.0},
        {"eps": 0.0},
    ],
)
def test_ccar3_invalid_params_rejected(
    kwargs: dict[str, object], two_views: list[np.ndarray]
) -> None:
    """Each of CCAR3's declared constraints rejects an out-of-range value."""
    with pytest.raises(InvalidParameterError):
        CCAR3(**kwargs).fit(two_views)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Valid values still work (regression guard against overly strict constraints)
# ---------------------------------------------------------------------------


def test_valid_parameters_still_fit(two_views: list[np.ndarray]) -> None:
    """Documented-valid parameter values are accepted, not false-positives."""
    rCCA(c=0.5).fit(two_views)
    MCCA(c=[0.1, 0.9], pca=False, eps=1e-8).fit(two_views)
    CCAR3(lambda_=0.1, highdim=False, rho=2.0, max_iter=100, tol=1e-3).fit(two_views)
