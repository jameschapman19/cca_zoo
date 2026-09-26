"""Tests for cca_zoo.linear._trimmed_cca (robust CCA via concentration steps)."""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo.linear import MCCA, RANSACCCA, TrimmedCCA
from cca_zoo.linear._trimmed_cca import _per_sample_terms, _select
from cca_zoo.metrics import average_pairwise_correlations, pairwise_correlations


def _make_model(**kwargs: object) -> TrimmedCCA:
    return TrimmedCCA(n_starts=3, max_iter=10, random_state=0, **kwargs)


# ---------------------------------------------------------------------------
# _select against brute force
# ---------------------------------------------------------------------------


def _brute_force_select(zs: list[np.ndarray], b: float, c: float, h: int) -> float:
    """Best (lowest) objective over every size-h subset, for small n only."""
    from itertools import combinations

    n = zs[0].shape[0]
    sigma, e, k_coef = _per_sample_terms(zs, b, c, h)
    best = np.inf
    for subset in combinations(range(n), h):
        idx = np.array(subset)
        val = sigma[idx].sum() + k_coef * e[idx].sum() ** 2
        best = min(best, val)
    return best


@pytest.mark.parametrize("m", [2, 3, 4], ids=["2-view", "3-view", "4-view"])
def test_select_usually_matches_brute_force_objective(m: int) -> None:
    """_select's chosen subset is exact in the large majority of trials.

    The Lagrangian relaxation solves ``h(mu) = 2*K*sum_{s in S(mu)} e(s) -
    mu = 0`` by bisection, which is exact whenever a root exists. At a
    ranking tie between two candidate subsets, though, ``h(mu)`` can jump
    straight across zero (a genuine integrality gap: no ``mu`` makes both
    sides of the tie self-consistent at once), and bisection then lands
    on the losing side of that jump -- occasionally landing on a subset
    that's meaningfully worse, not just off by a numerical sliver. This
    is why ``fit()`` never trusts ``_select`` alone: its own safeguard
    (reject a concentration step that doesn't actually lower CCAEY's real
    objective) is the thing that keeps the overall algorithm monotone,
    not any per-call guarantee from ``_select`` in isolation -- see
    ``test_trimmed_cca_beats_ransac_near_breakdown_point`` for that
    safeguard holding up end to end. Parametrized over the number of
    views ``m`` since the underlying algebra (and hence this guarantee)
    doesn't depend on it -- only on ``latent_dimensions == 1``.
    """
    rng = np.random.default_rng(0)
    n, h = 9, 5
    gaps = []
    for _ in range(60):
        zs = [rng.standard_normal(n) for _ in range(m)]
        b, c = 0.3, 0.1
        chosen = _select(zs, b, c, h)
        sigma, e, k_coef = _per_sample_terms(zs, b, c, h)
        chosen_val = sigma[chosen].sum() + k_coef * e[chosen].sum() ** 2
        brute_val = _brute_force_select(zs, b, c, h)
        gaps.append(chosen_val - brute_val)

    gaps_arr = np.array(gaps)
    n_exact = (gaps_arr <= 1e-8).sum()
    n_total = len(gaps_arr)
    assert n_exact >= 0.8 * n_total, f"only {n_exact}/{n_total} trials exact"


def test_select_returns_h_sorted_indices() -> None:
    """_select returns exactly h sorted, unique indices."""
    rng = np.random.default_rng(1)
    zs = [rng.standard_normal(20), rng.standard_normal(20)]
    chosen = _select(zs, b=0.2, c=0.1, h=12)
    assert chosen.shape == (12,)
    assert len(set(chosen.tolist())) == 12
    assert np.array_equal(chosen, np.sort(chosen))


# ---------------------------------------------------------------------------
# fit completes / shapes / validation
# ---------------------------------------------------------------------------


def test_two_view_fit_completes(two_views_small: list[np.ndarray]) -> None:
    """Fit completes on two-view data without error."""
    model = _make_model()
    fitted = model.fit(two_views_small)
    assert fitted is model


def test_three_view_fit_completes(three_views_small: list[np.ndarray]) -> None:
    """Fit completes on three-view data without error."""
    model = _make_model()
    fitted = model.fit(three_views_small)
    assert fitted is model
    assert len(fitted.weights_) == 3


def test_single_view_raises(two_views_small: list[np.ndarray]) -> None:
    """Fewer than 2 views is rejected (by the shared validate_views check)."""
    with pytest.raises(ValueError, match="At least 2 views"):
        _make_model().fit([two_views_small[0]])


def test_latent_dimensions_above_one_raises(two_views_small: list[np.ndarray]) -> None:
    """latent_dimensions > 1 is rejected explicitly."""
    with pytest.raises(ValueError, match="latent_dimensions=1"):
        _make_model(latent_dimensions=2).fit(two_views_small)


def test_weights_shapes_and_matches_transform(
    two_views_small: list[np.ndarray],
) -> None:
    """Weights are real (p_i, 1) arrays and transform(v) == centred(v) @ weights."""
    model = _make_model().fit(two_views_small)
    weights = model.weights_
    assert len(weights) == 2
    for w, v in zip(weights, two_views_small):
        assert w.shape == (v.shape[1], 1)

    transformed = model.transform(two_views_small)
    for v, w, t, mean in zip(two_views_small, weights, transformed, model.means_):
        np.testing.assert_allclose((v - mean) @ w, t, atol=1e-8)


def test_weights_not_fitted_raises() -> None:
    """Transform before fitting raises NotFittedError."""
    from sklearn.exceptions import NotFittedError

    model = TrimmedCCA()
    with pytest.raises(NotFittedError):
        model.transform([np.ones((3, 2)), np.ones((3, 2))])


def test_inlier_mask_shape(two_views_small: list[np.ndarray]) -> None:
    """inlier_mask_ is a boolean mask over the training samples, summing to h."""
    n = two_views_small[0].shape[0]
    model = _make_model(h_frac=0.6).fit(two_views_small)
    assert model.inlier_mask_.shape == (n,)
    assert model.inlier_mask_.dtype == bool
    assert model.inlier_mask_.sum() == max(2, round(0.6 * n))


# ---------------------------------------------------------------------------
# Correctness on clean data
# ---------------------------------------------------------------------------


def test_trimmed_cca_finds_correlation_on_correlated_views(
    correlated_views: list[np.ndarray],
) -> None:
    """On clean, uncontaminated data TrimmedCCA still finds real correlation."""
    model = TrimmedCCA(n_starts=5, random_state=0)
    s = model.fit(correlated_views).score(correlated_views)
    assert np.all(s > 0.5), f"Expected substantial correlation, got {s}"


# ---------------------------------------------------------------------------
# The headline scenario: near the ~50% breakdown point, TrimmedCCA holds up
# where RANSACCCA's minimal-random-subset search degrades.
# ---------------------------------------------------------------------------


def _make_sign_flip_contaminated_data(
    seed: int, n_train: int, n_test: int, p1: int, p2: int, contam_frac: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two views sharing one real latent factor, a fraction sign-flipped.

    See ``tests/test_ransac_cca.py``'s identical construction: nothing
    about a contaminated row's magnitude gives it away in either view, so
    this specifically isolates each method's ability to detect a wrong
    *relationship*, not an outlying magnitude.
    """
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal(p1)
    w1 /= np.linalg.norm(w1)
    w2 = rng.standard_normal(p2)
    w2 /= np.linalg.norm(w2)

    def group(n: int, sign: float) -> tuple[np.ndarray, np.ndarray]:
        t = rng.standard_normal(n)
        x = np.outer(t, w1) + 0.6 * rng.standard_normal((n, p1))
        y = np.outer(sign * t, w2) + 0.6 * rng.standard_normal((n, p2))
        return x, y

    x_test, y_test = group(n_test, sign=1.0)
    n_bad = int(round(contam_frac * n_train))
    n_good = n_train - n_bad
    x_good, y_good = group(n_good, sign=1.0)
    x_bad, y_bad = group(n_bad, sign=-1.0)
    x_train = np.vstack([x_good, x_bad])
    y_train = np.vstack([y_good, y_bad])
    contaminated = np.zeros(n_train, dtype=bool)
    contaminated[n_good:] = True

    perm = rng.permutation(n_train)
    return x_train[perm], y_train[perm], x_test, y_test, contaminated[perm]


def _held_out_corr(
    model: MCCA | RANSACCCA | TrimmedCCA, x_test: np.ndarray, y_test: np.ndarray
) -> float:
    z1, z2 = model.transform([x_test, y_test])
    return abs(np.corrcoef(z1[:, 0], z2[:, 0])[0, 1])


def _make_sign_flip_contaminated_data_multiview(
    seed: int, n_train: int, n_test: int, ps: list[int], contam_frac: float
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """M views sharing one latent factor; view 1 sign-flipped for bad rows.

    Generalises :func:`_make_sign_flip_contaminated_data` to any number of
    views: only the second view's sign is flipped for contaminated rows,
    so two of the ``M`` pairwise relationships break while the rest stay
    intact -- enough to hurt plain (non-robust) MCCA while still leaving
    something for a robust method to find.
    """
    rng = np.random.default_rng(seed)
    ws = [rng.standard_normal(p) for p in ps]
    ws = [w / np.linalg.norm(w) for w in ws]

    def group(n: int, sign: float) -> list[np.ndarray]:
        t = rng.standard_normal(n)
        out = []
        for i, (p, w) in enumerate(zip(ps, ws)):
            s = sign if i == 1 else 1.0
            out.append(np.outer(s * t, w) + 0.6 * rng.standard_normal((n, p)))
        return out

    test_views = group(n_test, sign=1.0)
    n_bad = int(round(contam_frac * n_train))
    n_good = n_train - n_bad
    good_views = group(n_good, sign=1.0)
    bad_views = group(n_bad, sign=-1.0)
    train_views = [np.vstack([g, b]) for g, b in zip(good_views, bad_views)]
    perm = rng.permutation(n_train)
    train_views = [v[perm] for v in train_views]
    return train_views, test_views, np.arange(n_train)[perm] >= n_good


def test_trimmed_cca_multiview_beats_plain_mcca() -> None:
    """With 3 views and 35% contamination, TrimmedCCA clearly beats plain MCCA.

    Regression coverage for the multiview generalisation: not just that
    ``fit`` completes on more than 2 views (see
    ``test_three_view_fit_completes``), but that the concentration-step
    algorithm actually recovers the real signal better than a non-robust
    baseline when more than 2 views are involved.
    """
    train_views, test_views, _ = _make_sign_flip_contaminated_data_multiview(
        seed=0, n_train=300, n_test=200, ps=[8, 6, 5], contam_frac=0.40
    )

    mcca_model = MCCA(latent_dimensions=1, c=0.1).fit(train_views)
    trimmed_model = TrimmedCCA(
        c=0.1, h_frac=0.63, n_starts=15, max_iter=30, random_state=0
    ).fit(train_views)

    mcca_corrs = average_pairwise_correlations(
        pairwise_correlations(mcca_model.transform(test_views))
    )
    trimmed_corrs = average_pairwise_correlations(
        pairwise_correlations(trimmed_model.transform(test_views))
    )

    assert trimmed_corrs[0] > abs(mcca_corrs[0]) + 0.08


def test_trimmed_cca_beats_ransac_near_breakdown_point() -> None:
    """Near ~47% contamination, TrimmedCCA (h_frac matched) beats RANSACCCA.

    RANSACCCA's random min_samples-sized draws become close to a coin
    flip on being usably clean once contamination approaches 50%, so its
    "try many random small subsets, keep the best-scoring one" search
    degrades regardless of tuning. TrimmedCCA's concentration steps
    don't depend on a lucky draw -- they start from a large h-sized
    subset and locally refine toward the best-scoring partition, so with
    h_frac set close to the true clean fraction they hold up where
    RANSACCCA's search does not.
    """
    contam_frac = 0.47
    x_train, y_train, x_test, y_test, _ = _make_sign_flip_contaminated_data(
        seed=0, n_train=400, n_test=300, p1=8, p2=6, contam_frac=contam_frac
    )

    ransac_corr = _held_out_corr(
        RANSACCCA(latent_dimensions=1, c=0.1, random_state=0).fit([x_train, y_train]),
        x_test,
        y_test,
    )
    trimmed_corr = _held_out_corr(
        TrimmedCCA(
            c=0.1,
            h_frac=1.0 - contam_frac + 0.02,
            n_starts=40,
            max_iter=30,
            random_state=0,
        ).fit([x_train, y_train]),
        x_test,
        y_test,
    )

    assert trimmed_corr > ransac_corr + 0.15


# ---------------------------------------------------------------------------
# sklearn compatibility spot-checks
# ---------------------------------------------------------------------------


def test_clone_and_get_params_roundtrip() -> None:
    """clone()/get_params() round-trip correctly (sklearn BaseEstimator contract)."""
    from sklearn.base import clone

    model = TrimmedCCA(c=0.2, h_frac=0.8, n_starts=5, max_iter=20, random_state=0)
    cloned = clone(model)
    assert cloned.get_params() == model.get_params()


def test_invalid_h_frac_raises() -> None:
    """h_frac outside (0, 1] is rejected by parameter validation."""
    from sklearn.utils._param_validation import InvalidParameterError

    with pytest.raises(InvalidParameterError):
        TrimmedCCA(h_frac=1.5)._validate_params()
    with pytest.raises(InvalidParameterError):
        TrimmedCCA(h_frac=0.0)._validate_params()


def test_invalid_n_starts_raises() -> None:
    """n_starts below 1 is rejected by parameter validation."""
    from sklearn.utils._param_validation import InvalidParameterError

    with pytest.raises(InvalidParameterError):
        TrimmedCCA(n_starts=0)._validate_params()
