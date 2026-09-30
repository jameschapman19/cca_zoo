"""The robust linear models, each against the contamination it is built for."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest
from scipy.optimize import approx_fprime

from cca_zoo._base import BaseModel
from cca_zoo.linear import (
    CCAEY,
    MCCA,
    RANSACCCA,
    HuberCCA,
    ProjectionPursuitCCA,
    TrimmedCCA,
)
from cca_zoo.linear._projection_pursuit_cca import spearman_projection_index
from cca_zoo.linear._trimmed_cca import _per_sample_terms, _select
from cca_zoo.linear.gradient._huber_cca import _huber_sample_weight
from cca_zoo.metrics import pairwise_correlations


def _contaminated(kind: str, fraction: float, noise: float = 0.6) -> tuple[list, list]:
    """Clean test views and training views with a fraction of bad rows.

    ``leverage``: a large, spuriously correlated cluster. ``sign``: ordinary
    rows whose second view follows the negated factor. ``outlier``: huge
    rows unrelated across views.
    """
    rng = np.random.default_rng(0)
    w = [rng.standard_normal(p) for p in (8, 6)]
    w = [v / np.linalg.norm(v) for v in w]

    def clean(n: int, sign: float = 1.0) -> list[np.ndarray]:
        t = rng.standard_normal(n)
        return [
            np.outer(s * t, v) + noise * rng.standard_normal((n, v.size))
            for s, v in zip((1.0, sign), w)
        ]

    test = clean(300)
    n_bad = int(fraction * 400)
    good = clean(400 - n_bad)
    if kind == "sign":
        bad = clean(n_bad, sign=-1.0)
    elif kind == "leverage":
        s = rng.standard_normal(n_bad)
        bad = [
            9.0 * np.outer(s, rng.standard_normal(v.size) / np.sqrt(v.size)) for v in w
        ]
    else:
        bad = [15.0 * rng.standard_normal((n_bad, v.size)) for v in w]
    order = rng.permutation(400)
    return [np.vstack(pair)[order] for pair in zip(good, bad)], test


def _held_out(model: BaseModel, train: list, test: list) -> float:
    z1, z2 = model.fit(train).transform(test)
    return abs(np.corrcoef(z1[:, 0], z2[:, 0])[0, 1])


@pytest.mark.parametrize(
    ("robust", "baseline", "kind", "fraction", "margin"),
    [
        (
            HuberCCA(max_iter=1500, random_state=0),
            CCAEY(max_iter=1500, random_state=0),
            "leverage",
            0.05,
            0.2,
        ),
        (
            RANSACCCA(min_samples=100, random_state=0),
            MCCA(shrinkage=0.1),
            "sign",
            0.4,
            0.3,
        ),
        (
            RANSACCCA(random_state=0),
            HuberCCA(max_iter=1500, random_state=0),
            "sign",
            0.4,
            0.3,
        ),
        (
            TrimmedCCA(
                shrinkage=0.1,
                support_fraction=0.55,
                n_init=40,
                max_iter=30,
                random_state=0,
            ),
            RANSACCCA(shrinkage=0.1, random_state=0),
            "sign",
            0.47,
            0.15,
        ),
        (
            ProjectionPursuitCCA(n_init=5, random_state=0),
            MCCA(shrinkage=0.1),
            "outlier",
            0.2,
            0.2,
        ),
    ],
    ids=[
        "Huber-leverage",
        "RANSAC-sign",
        "RANSAC-vs-Huber",
        "Trimmed-vs-RANSAC",
        "PP-outlier",
    ],
)
def test_resists_its_contamination(
    robust: BaseModel, baseline: BaseModel, kind: str, fraction: float, margin: float
) -> None:
    """On clean held-out data, the robust fit beats one the contamination misleads."""
    train, test = _contaminated(kind, fraction, noise=0.3 if kind == "outlier" else 0.6)
    assert _held_out(robust, train, test) > _held_out(baseline, train, test) + margin


@pytest.mark.parametrize("shrinkage", [0.0, 0.4])
def test_weighted_ey_gradient_matches_finite_differences(shrinkage: float) -> None:
    """With fixed sample weights, as HuberCCA's, CCAEY's gradient is exact."""
    rng = np.random.default_rng(0)
    views = [rng.standard_normal((15, p)) for p in (3, 4, 2)]
    sample_weight = rng.uniform(0.2, 1.0, size=15)

    class WeightedCCAEY(CCAEY):
        def _sample_weight(self, representations: list[np.ndarray]) -> np.ndarray:
            return sample_weight

    model = WeightedCCAEY(n_components=2, shrinkage=shrinkage)
    splits = np.cumsum([3 * 2, 4 * 2])

    def weights_of(flat: np.ndarray) -> list[np.ndarray]:
        return [part.reshape(-1, 2) for part in np.split(flat, splits)]

    def loss(flat: np.ndarray) -> float:
        weights = weights_of(flat)
        return model._objective(views, [v @ w for v, w in zip(views, weights)], weights)

    flat = rng.standard_normal(9 * 2)
    weights = weights_of(flat)
    gradient = model._derivative(
        views, [v @ w for v, w in zip(views, weights)], weights
    )
    np.testing.assert_allclose(
        np.concatenate([g.ravel() for g in gradient]),
        approx_fprime(flat, loss, 1e-6),
        atol=1e-5,
    )


def test_huber_keeps_half_the_batch_at_full_weight() -> None:
    """The cutoff is relative to the median leverage, so half the rows keep weight 1."""
    rng = np.random.default_rng(0)
    weight = _huber_sample_weight([rng.standard_normal((40, 2)) for _ in range(2)], 1.0)
    assert (weight >= 1.0 - 1e-9).sum() >= 20


def test_trimmed_subset_selection_is_near_optimal() -> None:
    """The relaxed subset choice is the brute-force optimum in most trials."""
    rng = np.random.default_rng(0)
    exact = 0
    for _ in range(60):
        zs = [rng.standard_normal(9) for _ in range(3)]
        sigma, e, k = _per_sample_terms(zs, 0.3, 0.1, 5)

        def cost(idx: np.ndarray) -> float:
            return sigma[idx].sum() + k * e[idx].sum() ** 2

        best = min(cost(np.array(s)) for s in combinations(range(9), 5))
        exact += cost(_select(zs, 0.3, 0.1, 5)) <= best + 1e-8
    assert exact >= 48


def test_trimmed_is_one_dimensional(two_views_small: list[np.ndarray]) -> None:
    """TrimmedCCA fits a single component."""
    with pytest.raises(ValueError, match="n_components"):
        TrimmedCCA(n_components=2).fit(two_views_small)


def test_spearman_index_is_rank_correlation() -> None:
    """The Spearman index is 1 for any monotone relationship, 0 for a constant."""
    u = np.arange(20.0)
    assert spearman_projection_index(u, u**3) == pytest.approx(1.0)
    assert spearman_projection_index(np.zeros(10), np.arange(10.0)) == 0.0


def test_projection_pursuit_components_track_cca_on_gaussian_data() -> None:
    """On Gaussian data rank correlation tracks Pearson's, component by component."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal((300, 2)) * [3, 1.5]
    views = [
        z @ rng.standard_normal((2, p)) + rng.standard_normal((300, p)) for p in (4, 3)
    ]
    robust = ProjectionPursuitCCA(n_components=2, n_init=3, random_state=0)
    np.testing.assert_allclose(
        pairwise_correlations(robust.fit(views).transform(views))[0, 1],
        pairwise_correlations(MCCA(n_components=2).fit(views).transform(views))[0, 1],
        atol=0.02,
    )
