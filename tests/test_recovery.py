"""Every model finds the signal the views share, not the loudest signal in one view.

Each view has a private factor three times the scale of the one factor they
share. Only the shared factor correlates across views, so any model of the
views' correlation must find it; a model of each view's own variance finds the
private factor instead.
"""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo._base import BaseModel
from tests._helpers import MODEL_CLASSES, fit, make_model, model_params

_N = 400

_DIAGONAL = (
    "it takes each view's covariance as diagonal, as PLS does, so its scores "
    "keep the private variance along the shared loading"
)
_EXEMPT = {
    "PLS": _DIAGONAL,
    "PLSEY": _DIAGONAL,
    "PMDCCA": _DIAGONAL,
    "SpanCCA": _DIAGONAL,
    "ParkhomenkoCCA": _DIAGONAL,
    "GFA": "its latent dimensions model private factors too, by design",
    "ManifoldCCA": (
        "each view's embedding is built from that view's neighbourhood graph, "
        "which here follows the private factor"
    ),
}

# Settings under which a model can find the shared factor at all.
_SETTINGS = {
    "ProbabilisticCCA": {"n_warmup": 300, "n_posterior_samples": 300},
    "VariationalBayesCCA": {"n_iter": 3000},
    # Cancelling the private factor needs two features.
    "OrthogonalMatchingPursuitCCA": {"n_nonzero_coefs": 2},
}

# Models whose quick test settings stop them short: these run at their defaults.
_AT_DEFAULTS = {
    "XGBoostCCA",
    "LightGBMCCA",
    "CatBoostCCA",
    "ProjectionPursuitCCA",
    "StochasticCCAEY",
}

# Piecewise-constant fits approximate the linear signal.
_TREES = {"XGBoostCCA", "LightGBMCCA", "CatBoostCCA"}


def _views() -> tuple[list[np.ndarray], np.ndarray]:
    """Two views of one shared factor, each dominated by its own private factor."""
    rng = np.random.default_rng(0)
    shared = rng.standard_normal((_N, 1))
    views = [
        3 * rng.standard_normal((_N, 1)) @ np.ones((1, p))
        + shared @ rng.standard_normal((1, p))
        + 0.3 * rng.standard_normal((_N, p))
        for p in (6, 5)
    ]
    return views, shared[:, 0]


@pytest.mark.parametrize(
    "cls", model_params([c for c in MODEL_CLASSES if c.__name__ not in _EXEMPT])
)
def test_the_first_component_is_the_shared_factor(cls: type[BaseModel]) -> None:
    """Each view's first score follows the shared factor."""
    views, shared = _views()
    model = cls() if cls.__name__ in _AT_DEFAULTS else make_model(cls)
    if "random_state" in model.get_params():
        model.set_params(random_state=0)
    model.set_params(**_SETTINGS.get(cls.__name__, {}))
    for scores in fit(model, views).transform(views):
        correlation = abs(np.corrcoef(np.asarray(scores)[:, 0], shared)[0, 1])
        assert correlation > (0.8 if cls.__name__ in _TREES else 0.9)
