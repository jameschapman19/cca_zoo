"""The Eckart-Young gradient models, checked against the closed forms."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import approx_fprime
from sklearn.exceptions import ConvergenceWarning

from cca_zoo._utils._ey import ey_loss
from cca_zoo.linear import CCA, CCAEY, MCCA, PLS, PLSEY, HuberCCA
from cca_zoo.stochastic import StochasticCCAEY
from tests._helpers import canonical_correlations, linear_views, ordered_views


def _views(n_views: int) -> list[np.ndarray]:
    return linear_views(0, 300, (10, 8, 6)[:n_views], noise=0.1)


@pytest.mark.parametrize(
    ("model", "exact", "n_views"),
    [
        (CCAEY(n_components=2, random_state=0), CCA(n_components=2), 2),
        (PLSEY(n_components=2, random_state=0), PLS(n_components=2), 2),
        (CCAEY(n_components=2, random_state=0), MCCA(n_components=2), 3),
        (StochasticCCAEY(n_components=2, random_state=0), CCA(n_components=2), 2),
        # 300 rows in batches of 299 leave a remainder of one row.
        (
            StochasticCCAEY(n_components=2, batch_size=299, random_state=0),
            CCA(n_components=2),
            2,
        ),
    ],
    ids=[
        "CCAEY-CCA",
        "PLSEY-PLS",
        "CCAEY-MCCA",
        "Stochastic-CCA",
        "Stochastic-one-row-remainder",
    ],
)
def test_converges_to_the_closed_form(
    model: object, exact: object, n_views: int
) -> None:
    """The EY optimum has the closed-form solution's correlations, in some order."""
    views = _views(n_views)
    np.testing.assert_allclose(
        np.sort(canonical_correlations(model.fit(views), views)),
        np.sort(canonical_correlations(exact.fit(views), views)),
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


def test_stochastic_batch_smaller_than_n_components_fits() -> None:
    """A batch of fewer rows than components initialises and finds the signal.

    The start whitens one mini-batch with a QR, whose factor is not square when
    the batch has fewer rows than components. The views share two strong
    components, which the fit recovers. The step is small because batches this
    small diverge at the default one for most seeds.
    """
    views = _views(2)
    model = StochasticCCAEY(
        n_components=8,
        batch_size=6,
        max_iter=200,
        learning_rate=0.05,
        random_state=0,
    ).fit(views)
    exact = CCA(n_components=2).fit(views)
    np.testing.assert_allclose(
        np.sort(canonical_correlations(model, views))[::-1][:2],
        np.sort(canonical_correlations(exact, views))[::-1],
        atol=0.1,
    )


@pytest.mark.parametrize("cls", [CCAEY, PLSEY, HuberCCA])
def test_ignores_the_units_of_a_view(cls: type) -> None:
    """Rescaling each view leaves CCA's and PLS's correlations unchanged."""
    views = _views(2)
    rescaled = [views[0] * 100, views[1] * 0.01]
    np.testing.assert_allclose(
        canonical_correlations(
            cls(n_components=2, random_state=0).fit(rescaled), rescaled
        ),
        canonical_correlations(cls(n_components=2, random_state=0).fit(views), views),
        atol=1e-3,
    )


def test_stochastic_full_batch_reaches_cca() -> None:
    """A stall shrinks the step rather than stopping short of CCA's optimum."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((101, 5))
    views = [x, x[:, :3] + rng.standard_normal((101, 3))]
    assert StochasticCCAEY(random_state=0).fit(views).score(views) == pytest.approx(
        CCA().fit(views).score(views), abs=0.01
    )


@pytest.mark.parametrize("shrinkage", [0.0, 0.4])
@pytest.mark.parametrize("ordered", [False, True])
def test_gradient_matches_finite_differences(ordered: bool, shrinkage: float) -> None:
    """The analytic gradient is the derivative of the objective, ordered or not."""
    rng = np.random.default_rng(0)
    views = [rng.standard_normal((40, p)) for p in (5, 4, 3)]
    views = [v - v.mean(axis=0) for v in views]
    model = CCAEY(n_components=3, ordered=ordered, shrinkage=shrinkage)
    shapes = [(v.shape[1], 3) for v in views]
    splits = np.cumsum([np.prod(s) for s in shapes])[:-1]

    def unpack(flat: np.ndarray) -> list[np.ndarray]:
        return [p.reshape(s) for p, s in zip(np.split(flat, splits), shapes)]

    def objective(flat: np.ndarray) -> float:
        w = unpack(flat)
        return model._objective(views, [v @ x for v, x in zip(views, w)], w)

    flat = 0.3 * rng.standard_normal(sum(int(np.prod(s)) for s in shapes))
    w = unpack(flat)
    gradient = np.concatenate(
        [
            g.ravel()
            for g in model._derivative(views, [v @ x for v, x in zip(views, w)], w)
        ]
    )
    np.testing.assert_allclose(
        gradient, approx_fprime(flat, objective, 1e-6), atol=1e-5
    )


def test_ordered_objective_is_the_mean_loss_over_prefixes() -> None:
    """At shrinkage 0 the ordered objective is the mean EY loss over prefixes."""
    rng = np.random.default_rng(0)
    scores = [rng.standard_normal((50, 4)) for _ in range(3)]
    by_definition = np.mean(
        [ey_loss([z[:, :p] for z in scores])["objective"] for p in range(1, 5)]
    )
    views = [np.zeros((50, 4)) for _ in scores]
    weights = [np.eye(4) for _ in scores]
    model = CCAEY(n_components=4, ordered=True)
    np.testing.assert_allclose(model._objective(views, scores, weights), by_definition)


def _alignment_with_cca(
    model: CCAEY | StochasticCCAEY, views: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per component, the correlation with CCA's variate of the same index."""
    ours = model.fit(views).transform(views)[0]
    theirs = CCA(n_components=ours.shape[1]).fit(views).transform(views)[0]
    ours, theirs = ours - ours.mean(axis=0), theirs - theirs.mean(axis=0)
    return (
        ours,
        theirs,
        np.abs((ours * theirs).sum(axis=0))
        / np.sqrt((ours**2).sum(axis=0) * (theirs**2).sum(axis=0)),
    )


@pytest.mark.parametrize(
    "model",
    [
        CCAEY(n_components=3, ordered=True, random_state=0),
        StochasticCCAEY(n_components=3, ordered=True, random_state=0),
    ],
    ids=["CCAEY", "StochasticCCAEY"],
)
def test_ordered_components_are_the_canonical_directions_in_order(
    model: CCAEY | StochasticCCAEY,
) -> None:
    """Component j correlates with CCA's j-th variate, and the span is CCA's."""
    views = ordered_views(0, 2000, (8, 6))
    ours, theirs, alignment = _alignment_with_cca(model, views)
    assert alignment.min() > 0.99
    beta = np.linalg.lstsq(ours, theirs, rcond=None)[0]
    assert np.all(
        ((theirs - ours @ beta) ** 2).sum(axis=0) < 0.01 * (theirs**2).sum(axis=0)
    )


def test_ordered_minibatch_fit_is_sorted_with_patience() -> None:
    """Mini-batches sort the components too, given the patience to do it.

    The default five stalled epochs shrink the step before the weakest
    direction has separated from its neighbour, as the noisy epoch loss rarely
    improves by ``tol`` there.
    """
    views = ordered_views(0, 2000, (8, 6))
    model = StochasticCCAEY(
        n_components=3,
        ordered=True,
        batch_size=256,
        n_iter_no_change=100,
        random_state=0,
    )
    with pytest.warns(ConvergenceWarning):
        _, _, alignment = _alignment_with_cca(model, views)
    assert alignment.min() > 0.9
