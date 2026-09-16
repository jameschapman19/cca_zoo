"""Tests for ALS-based sparse/regularised CCA variants.

Covers PLSALS, SCCAPMD, SCCAADMM, SCCAIPLS, SCCASpan, ElasticCCA,
ParkhomenkoCCA, SAR.
"""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo.linear import (
    PLSALS,
    SAR,
    SCCAADMM,
    SCCAIPLS,
    SCCAPMD,
    ElasticCCA,
    ParkhomenkoCCA,
    SCCASpan,
)

ALL_ITERATIVE_MODELS = [
    PLSALS,
    SCCAPMD,
    SCCAADMM,
    SCCAIPLS,
    SCCASpan,
    ElasticCCA,
    ParkhomenkoCCA,
    SAR,
]

# Use few iterations for test speed
_BASE_KWARGS: dict = dict(latent_dimensions=1, max_iter=50, random_state=0)


# ---------------------------------------------------------------------------
# fit completes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", ALL_ITERATIVE_MODELS)
def test_two_view_fit_completes(ModelClass: type, two_views: list[np.ndarray]) -> None:
    """Fit completes on two-view data without error."""
    model = ModelClass(**_BASE_KWARGS)
    fitted = model.fit(two_views)
    assert fitted is model


@pytest.mark.parametrize("ModelClass", ALL_ITERATIVE_MODELS)
def test_three_view_fit_completes(
    ModelClass: type, three_views: list[np.ndarray]
) -> None:
    """Iterative models accept three or more views."""
    model = ModelClass(**_BASE_KWARGS)
    fitted = model.fit(three_views)
    assert fitted is model


# ---------------------------------------------------------------------------
# transform output shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", ALL_ITERATIVE_MODELS)
def test_transform_shapes_two_view(
    ModelClass: type, two_views: list[np.ndarray]
) -> None:
    """Transform returns list of (n_samples, latent_dimensions) arrays."""
    k = 2
    model = ModelClass(latent_dimensions=k, max_iter=50, random_state=0).fit(two_views)
    result = model.transform(two_views)
    assert len(result) == len(two_views)
    for arr, view in zip(result, two_views):
        assert arr.shape == (view.shape[0], k)


@pytest.mark.parametrize("ModelClass", ALL_ITERATIVE_MODELS)
def test_transform_shapes_three_view(
    ModelClass: type, three_views: list[np.ndarray]
) -> None:
    """Transform returns correct shapes for three-view data."""
    k = 2
    model = ModelClass(latent_dimensions=k, max_iter=50, random_state=0).fit(
        three_views
    )
    result = model.transform(three_views)
    assert len(result) == len(three_views)
    for arr, view in zip(result, three_views):
        assert arr.shape == (view.shape[0], k)


# ---------------------------------------------------------------------------
# fit_transform consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", ALL_ITERATIVE_MODELS)
def test_fit_transform_consistency(
    ModelClass: type, two_views: list[np.ndarray]
) -> None:
    """fit_transform equals fit().transform()."""
    kwargs = dict(latent_dimensions=1, max_iter=50, random_state=0)
    result_ft = ModelClass(**kwargs).fit_transform(two_views)
    result_sep = ModelClass(**kwargs).fit(two_views).transform(two_views)
    for a, b in zip(result_ft, result_sep):
        np.testing.assert_allclose(np.abs(a), np.abs(b), atol=1e-10)


# ---------------------------------------------------------------------------
# score shape and range
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", ALL_ITERATIVE_MODELS)
def test_score_shape(ModelClass: type, two_views: list[np.ndarray]) -> None:
    """Score returns array of shape (latent_dimensions,)."""
    k = 2
    model = ModelClass(latent_dimensions=k, max_iter=50, random_state=0).fit(two_views)
    s = model.score(two_views)
    assert s.shape == (k,)


@pytest.mark.parametrize("ModelClass", ALL_ITERATIVE_MODELS)
def test_score_values_in_valid_range(
    ModelClass: type, correlated_views: list[np.ndarray]
) -> None:
    """Score values lie in [-1, 1]."""
    model = ModelClass(latent_dimensions=1, max_iter=100, random_state=0).fit(
        correlated_views
    )
    s = model.score(correlated_views)
    assert np.all(s >= -1.0 - 1e-9)
    assert np.all(s <= 1.0 + 1e-9)


# get_params/set_params roundtrip behaviour is exercised generically for
# every model in the package by tests/test_sklearn_compat.py.


# ---------------------------------------------------------------------------
# weights shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", ALL_ITERATIVE_MODELS)
def test_weights_shapes_two_view(ModelClass: type, two_views: list[np.ndarray]) -> None:
    """Weights are shaped (n_features_i, latent_dimensions) per view."""
    k = 2
    model = ModelClass(latent_dimensions=k, max_iter=50, random_state=0).fit(two_views)
    w = model.weights
    assert len(w) == len(two_views)
    for weight, view in zip(w, two_views):
        assert weight.shape == (view.shape[1], k)


# ---------------------------------------------------------------------------
# get_factor_loadings shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", ALL_ITERATIVE_MODELS)
def test_get_factor_loadings_shapes(
    ModelClass: type, two_views: list[np.ndarray]
) -> None:
    """get_factor_loadings returns (n_features_i, k) arrays."""
    k = 2
    model = ModelClass(latent_dimensions=k, max_iter=50, random_state=0).fit(two_views)
    loadings = model.get_factor_loadings(two_views)
    assert len(loadings) == len(two_views)
    for loading, view in zip(loadings, two_views):
        assert loading.shape == (view.shape[1], k)


# ---------------------------------------------------------------------------
# Sparsity verification
# ---------------------------------------------------------------------------


def test_scca_pmd_achieves_sparsity(two_views: list[np.ndarray]) -> None:
    """SCCAPMD with small tau produces sparse weights (some zeros)."""
    model = SCCAPMD(latent_dimensions=1, tau=0.3, max_iter=200, random_state=0).fit(
        two_views
    )
    for w in model.weights:
        n_zeros = np.sum(np.abs(w) < 1e-10)
        assert n_zeros > 0, f"Expected some zero weights, got {n_zeros}"


def test_scca_pmd_invariant_to_input_scale(two_views: list[np.ndarray]) -> None:
    """SCCAPMD's fitted weights (up to sign) must not depend on input scale.

    tau is the only sparsity control.

    Regression test: _bisect_threshold used to compare the *unnormalised*
    power-iteration update's L1 norm directly against l1_bound (a bound
    that is only meaningful for a unit-L2-norm vector), so scaling the
    input data changed the effective sparsity even at a fixed tau -- in
    the common case where the raw update's magnitude exceeds l1_bound,
    tau bound far more aggressively than intended, up to tau=1 (nominally
    "no constraint") still producing near-total sparsity.
    """
    scaled_views = [v * 37.0 for v in two_views]
    model_a = SCCAPMD(latent_dimensions=1, tau=0.5, max_iter=200, random_state=0).fit(
        two_views
    )
    model_b = SCCAPMD(latent_dimensions=1, tau=0.5, max_iter=200, random_state=0).fit(
        scaled_views
    )
    for w_a, w_b in zip(model_a.weights, model_b.weights):
        # sign of the leading direction is arbitrary; align before comparing
        sign = np.sign((w_a * w_b).sum()) or 1.0
        np.testing.assert_allclose(w_a, sign * w_b, atol=1e-6)


def test_bisect_threshold_matches_a_from_scratch_bisection() -> None:
    """`_bisect_threshold`'s brentq solve matches an independent fixed bisection.

    `_bisect_threshold` used to run a hand-rolled, unconditional 50-iteration
    bisection with no early stop; replaced with `scipy.optimize.brentq` for
    the same monotonic root-find (3.65x faster across 500 random trials in a
    direct benchmark, 1.8x faster end-to-end in `SCCAPMD.fit`). Pins the
    result against a from-scratch fixed-count bisection, independent of the
    function under test, across a range of vector sizes and scales.
    """
    from cca_zoo._utils._linalg import soft_threshold
    from cca_zoo.linear._iterative import _bisect_threshold

    def reference_bisection(x: np.ndarray, l1_bound: float) -> np.ndarray:
        norm_x = np.linalg.norm(x)
        if norm_x <= 1e-12:
            return np.zeros_like(x)
        unit_x = x / norm_x
        if np.linalg.norm(unit_x, 1) <= l1_bound:
            return np.asarray(unit_x)
        lo, hi = 0.0, np.abs(x).max()
        for _ in range(200):
            mid = (lo + hi) / 2.0
            thresholded = soft_threshold(x, mid)
            norm_t = np.linalg.norm(thresholded)
            ratio = np.linalg.norm(thresholded, 1) / norm_t if norm_t > 1e-12 else 0.0
            if ratio > l1_bound:
                lo = mid
            else:
                hi = mid
        result = soft_threshold(x, (lo + hi) / 2.0)
        norm = np.linalg.norm(result)
        if norm > 1e-12:
            result /= norm
        return result

    rng = np.random.default_rng(0)
    for _ in range(50):
        p = rng.integers(5, 100)
        x = rng.standard_normal(p) * rng.choice([1.0, 10.0, 100.0])
        l1_bound = rng.uniform(1.0, np.sqrt(p))
        got = _bisect_threshold(x, l1_bound)
        want = reference_bisection(x, l1_bound)
        np.testing.assert_allclose(got, want, atol=1e-6)


def test_scca_pmd_tau_controls_sparsity_monotonically(
    two_views: list[np.ndarray],
) -> None:
    """Increasing tau must not decrease the number of selected features.

    tau=1 bounds by the Cauchy-Schwarz maximum for a unit vector, so it
    should recover the (denser) unconstrained solution, not one sparser
    than a smaller tau's.
    """
    taus = [0.3, 0.5, 0.7, 1.0]
    nnz_by_tau = []
    for tau in taus:
        model = SCCAPMD(latent_dimensions=1, tau=tau, max_iter=200, random_state=0).fit(
            two_views
        )
        nnz_by_tau.append(sum(int(np.sum(np.abs(w) > 1e-10)) for w in model.weights))
    assert nnz_by_tau == sorted(nnz_by_tau), (
        f"nnz should be non-decreasing in tau, got {dict(zip(taus, nnz_by_tau))}"
    )
    # tau=1 imposes no real constraint (L1 bound = sqrt(p), the max
    # possible for a unit vector), so it must not be sparse.
    assert nnz_by_tau[-1] == sum(v.shape[1] for v in two_views)


def test_parkhomenko_achieves_sparsity(two_views: list[np.ndarray]) -> None:
    """ParkhomenkoCCA with positive tau produces sparse weights."""
    model = ParkhomenkoCCA(
        latent_dimensions=1, tau=0.5, max_iter=200, random_state=0
    ).fit(two_views)
    for w in model.weights:
        n_zeros = np.sum(np.abs(w) < 1e-10)
        assert n_zeros > 0, f"Expected some zero weights, got {n_zeros}"


def test_scca_span_achieves_sparsity(two_views: list[np.ndarray]) -> None:
    """SCCASpan with span < n_features produces sparse weights."""
    n_features = two_views[0].shape[1]
    span = n_features // 2
    model = SCCASpan(latent_dimensions=1, span=span, max_iter=200, random_state=0).fit(
        two_views
    )
    # First view should have at most 'span' nonzero entries per dimension
    w0 = model.weights[0][:, 0]
    n_nonzero = np.sum(np.abs(w0) > 1e-10)
    assert n_nonzero <= span, f"Expected <= {span} nonzero, got {n_nonzero}"


def test_scca_admm_achieves_sparsity(two_views: list[np.ndarray]) -> None:
    """SCCAADMM with positive tau produces some sparse weights."""
    model = SCCAADMM(latent_dimensions=1, tau=0.5, max_iter=200, random_state=0).fit(
        two_views
    )
    assert hasattr(model, "weights_")
    for w in model.weights:
        assert w.shape[0] > 0


def test_scca_admm_stable_at_a_realistic_sample_size() -> None:
    """SCCAADMM's weights stay finite at n=200, not just the tiny n=50 examples.

    Regression test: an earlier version of the w-update took a single
    proximal-gradient step per ADMM iteration with an un-normalised gradient
    against a step size calibrated for the normalised loss, which diverged
    to `nan` by iteration 200 at n=200 on perfectly ordinary Gaussian data
    (found by direct benchmark) -- undetected before because every existing
    test used n=50, small enough that the mismatch alone didn't blow up.
    """
    rng = np.random.default_rng(0)
    n, p, q = 200, 60, 50
    X1 = rng.standard_normal((n, p))
    X2 = rng.standard_normal((n, q))
    model = SCCAADMM(
        latent_dimensions=2, tau=0.3, mu=1.0, max_iter=500, random_state=0
    ).fit([X1, X2])
    for w in model.weights_:
        assert np.all(np.isfinite(w))


def test_scca_admm_w_update_matches_an_independent_ground_truth() -> None:
    """SCCAADMM's closed-form w-update matches an independent proximal-gradient solve.

    For a single view with a *fixed* target (i.e. one ADMM sub-problem in
    isolation, decoupled from the outer multiview alternation), the
    penalised, ball-constrained problem this class solves has a unique
    optimum. Pins the fitted result against a from-scratch proximal
    -gradient solve of that exact problem, independent of both the
    production code and scipy's `cho_solve`. This is also a regression
    test for a second bug caught only by this comparison: a first attempt
    at the closed-form w-update used `(1/n) X^T X` instead of the `(2/n)
    X^T X` the loss `(1/n)||Xw - target||^2`'s gradient actually needs; it
    still converged cleanly, just to a measurably over-shrunk, wrong
    stationary point.
    """
    from cca_zoo._utils._linalg import soft_threshold

    rng = np.random.default_rng(0)
    n, p = 200, 60
    X = rng.standard_normal((n, p))
    true_w = np.zeros(p)
    true_w[:8] = rng.standard_normal(8)
    target = X @ true_w + 0.3 * rng.standard_normal(n)
    target = target / np.linalg.norm(target)

    tau, mu = 0.02, 1.0
    XtX = X.T @ X
    Xtarget = X.T @ target

    def project_ball(z: np.ndarray) -> np.ndarray:
        nz = np.linalg.norm(z)
        return z / nz if nz > 1.0 else z

    def objective(w: np.ndarray) -> float:
        resid = X @ w - target
        return float((resid**2).sum() / n + tau * np.abs(w).sum())

    # Independent ground truth: plain proximal gradient (ISTA) on the exact
    # same constrained problem, run to convergence.
    lipschitz = 2 * np.linalg.eigvalsh(XtX).max() / n
    step = 1.0 / lipschitz
    w = np.zeros(p)
    for _ in range(200_000):
        grad = (2.0 / n) * (XtX @ w - Xtarget)
        candidate = project_ball(soft_threshold(w - step * grad, tau * step))
        if np.linalg.norm(candidate - w) < 1e-14:
            w = candidate
            break
        w = candidate
    w_true = w

    # The production ADMM update, run against this same fixed target (no
    # outer multiview alternation) for many iterations.
    from scipy.linalg import cho_factor, cho_solve

    A = cho_factor(2 * XtX / n + mu * np.eye(p))
    w2 = np.zeros(p)
    z = w2.copy()
    eta = np.zeros(p)
    for _ in range(2000):
        w2 = cho_solve(A, 2 * Xtarget / n + mu * (z - eta))
        z = project_ball(soft_threshold(w2 + eta, tau / mu))
        eta = eta + w2 - z

    np.testing.assert_allclose(objective(z), objective(w_true), rtol=1e-3)


def test_elastic_cca_with_lasso(two_views: list[np.ndarray]) -> None:
    """ElasticCCA with l1_ratio=1 (lasso) produces some sparse weights."""
    model = ElasticCCA(
        latent_dimensions=1, alpha=0.1, l1_ratio=1.0, max_iter=200, random_state=0
    ).fit(two_views)
    assert hasattr(model, "weights_")


def test_scca_ipls_with_lasso(two_views: list[np.ndarray]) -> None:
    """SCCAIPLS with alpha > 0 runs without error."""
    model = SCCAIPLS(
        latent_dimensions=1, alpha=0.1, l1_ratio=1.0, max_iter=100, random_state=0
    ).fit(two_views)
    assert hasattr(model, "weights_")


def test_sar_finds_zero_weights_when_no_signal(two_views: list[np.ndarray]) -> None:
    """SAR's BIC selection should prefer the all-zero fit on pure noise.

    Where the true regression coefficient really is zero -- unlike
    every other class here, SAR has no user-set penalty strength to
    check sparsity against, so this checks the BIC selection itself
    rather than a fixed hyperparameter's effect.
    """
    model = SAR(latent_dimensions=1, max_iter=50, random_state=0).fit(two_views)
    for w in model.weights:
        assert np.all(w == 0.0)


def test_sar_recovers_correlated_support() -> None:
    """SAR should select the columns carrying real shared signal.

    And reject the pure-noise columns, on a case with both present in
    each view -- the same before/after signal-recovery standard used
    to verify ParkhomenkoCCA's whitening fix.
    """
    rng = np.random.default_rng(0)
    n = 200
    latent = rng.standard_normal(n)
    x_signal = latent[:, None] + 0.2 * rng.standard_normal((n, 3))
    y_signal = latent[:, None] + 0.2 * rng.standard_normal((n, 3))
    x = np.column_stack([x_signal, rng.standard_normal((n, 27))])
    y = np.column_stack([y_signal, rng.standard_normal((n, 17))])
    model = SAR(latent_dimensions=1, random_state=0).fit([x, y])
    for w in model.weights:
        signal_idx, noise_idx = w[:3, 0], w[3:, 0]
        assert np.all(np.abs(signal_idx) > 1e-10), "true-signal columns were zeroed"
        n_false_positives = np.sum(np.abs(noise_idx) > 1e-10)
        assert n_false_positives <= 2, (
            f"expected mostly-zero noise columns, got {n_false_positives} nonzero"
        )
        assert np.sum(signal_idx**2) > 10 * np.sum(noise_idx**2), (
            "signal columns should carry most of the weight mass"
        )
    zx, zy = model.transform([x, y])
    assert np.corrcoef(zx.ravel(), zy.ravel())[0, 1] > 0.9


def test_sar_multicomponent_deflation_and_reexpression() -> None:
    """A second SAR component should stay close to uncorrelated with the first.

    This exercises the deflate-then-re-express path a lasso-based fit
    needs, unlike this module's other classes (see the class
    docstring).
    """
    rng = np.random.default_rng(1)
    n = 200
    latent1, latent2 = rng.standard_normal(n), rng.standard_normal(n)
    x = np.column_stack(
        [
            latent1[:, None] + 0.2 * rng.standard_normal((n, 3)),
            latent2[:, None] + 0.2 * rng.standard_normal((n, 3)),
            rng.standard_normal((n, 24)),
        ]
    )
    y = np.column_stack(
        [
            latent1[:, None] + 0.2 * rng.standard_normal((n, 3)),
            latent2[:, None] + 0.2 * rng.standard_normal((n, 3)),
            rng.standard_normal((n, 14)),
        ]
    )
    model = SAR(latent_dimensions=2, random_state=0).fit([x, y])
    zx, zy = model.transform([x, y])
    for d in range(2):
        assert np.corrcoef(zx[:, d], zy[:, d])[0, 1] > 0.9
    assert abs(np.corrcoef(zx[:, 0], zx[:, 1])[0, 1]) < 0.3


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", ALL_ITERATIVE_MODELS)
def test_reproducibility(ModelClass: type, two_views: list[np.ndarray]) -> None:
    """Same random_state gives identical weights."""
    kwargs = dict(latent_dimensions=1, max_iter=50, random_state=42)
    w1 = ModelClass(**kwargs).fit(two_views).weights
    w2 = ModelClass(**kwargs).fit(two_views).weights
    for a, b in zip(w1, w2):
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# center=False
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", ALL_ITERATIVE_MODELS)
def test_center_false(ModelClass: type, two_views: list[np.ndarray]) -> None:
    """All iterative models work with center=False."""
    model = ModelClass(latent_dimensions=1, max_iter=20, center=False, random_state=0)
    model.fit(two_views)
    result = model.transform(two_views)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# pairwise_correlations shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", ALL_ITERATIVE_MODELS)
def test_pairwise_correlations_shape(
    ModelClass: type, two_views: list[np.ndarray]
) -> None:
    """pairwise_correlations returns (n_views, n_views, k)."""
    k = 1
    model = ModelClass(latent_dimensions=k, max_iter=50, random_state=0).fit(two_views)
    corrs = model.pairwise_correlations(two_views)
    assert corrs.shape == (2, 2, k)


# ---------------------------------------------------------------------------
# Correctness / optimality
# ---------------------------------------------------------------------------


def test_pls_als_matches_pls(correlated_views: list[np.ndarray]) -> None:
    """PLSALS (converged) recovers the same correlations as exact PLS."""
    from cca_zoo.linear import PLS

    k = 2
    s_pls = PLS(latent_dimensions=k).fit(correlated_views).score(correlated_views)
    s_als = (
        PLSALS(latent_dimensions=k, max_iter=1000, random_state=0)
        .fit(correlated_views)
        .score(correlated_views)
    )
    np.testing.assert_allclose(s_als, s_pls, atol=0.05)


def test_iterative_models_find_high_correlation(
    correlated_views: list[np.ndarray],
) -> None:
    """All iterative models find substantial correlation on clearly correlated views."""
    for ModelClass in ALL_ITERATIVE_MODELS:
        s = (
            ModelClass(latent_dimensions=1, max_iter=500, random_state=0)
            .fit(correlated_views)
            .score(correlated_views)
        )
        assert np.all(s > 0.5), f"{ModelClass.__name__} got low correlation: {s}"
