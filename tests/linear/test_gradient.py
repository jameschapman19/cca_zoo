"""The Eckart-Young gradient models, checked against the closed forms."""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo.linear import CCA, CCAEY, MCCA, PLS, PLSEY
from cca_zoo.stochastic import StochasticCCAEY
from tests._helpers import canonical_correlations


def _views(n_views: int) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    z = rng.standard_normal((300, 2))
    return [
        z @ rng.standard_normal((2, p)) + 0.1 * rng.standard_normal((300, p))
        for p in (10, 8, 6)[:n_views]
    ]


@pytest.mark.parametrize(
    ("model", "exact", "n_views"),
    [
        (CCAEY(n_components=2, random_state=0), CCA(n_components=2), 2),
        (PLSEY(n_components=2, random_state=0), PLS(n_components=2), 2),
        (CCAEY(n_components=2, random_state=0), MCCA(n_components=2), 3),
        (
            StochasticCCAEY(
                n_components=2,
                batch_size=100,
                max_iter=300,
                random_state=0,
            ),
            CCA(n_components=2),
            2,
        ),
    ],
    ids=["CCAEY-CCA", "PLSEY-PLS", "CCAEY-MCCA", "Stochastic-CCA"],
)
def test_converges_to_the_closed_form(
    model: object, exact: object, n_views: int
) -> None:
    """The EY optimum has the closed-form solution's correlations, in order."""
    views = _views(n_views)
    np.testing.assert_allclose(
        canonical_correlations(model.fit(views), views),
        canonical_correlations(exact.fit(views), views),
        atol=0.05,
    )


def test_ccaey_with_full_shrinkage_is_plsey(two_views: list[np.ndarray]) -> None:
    """At shrinkage=1 CCAEY's objective and gradient are PLSEY's."""
    rng = np.random.default_rng(0)
    weights = [rng.standard_normal((v.shape[1], 2)) for v in two_views]
    scores = [v @ w for v, w in zip(two_views, weights)]
    pls, cca = PLSEY(n_components=2), CCAEY(n_components=2, shrinkage=1.0)
    assert pls._objective(two_views, scores, weights) == cca._objective(
        two_views, scores, weights
    )
    for a, b in zip(
        pls._derivative(two_views, scores, weights),
        cca._derivative(two_views, scores, weights),
    ):
        np.testing.assert_array_equal(a, b)


def test_stochastic_divergence_is_an_error() -> None:
    """Too large a step diverges, and says what to change."""
    with pytest.raises(ValueError, match="Lower learning_rate"):
        StochasticCCAEY(
            n_components=2, learning_rate=2.0, batch_size=100, random_state=0
        ).fit(_views(2))
