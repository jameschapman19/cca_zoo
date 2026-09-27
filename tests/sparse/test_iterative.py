"""Tests for ALS-based sparse/regularised CCA variants.

Covers PMDCCA, ADMMCCA, IPLSCCA, SpanCCA, WaijenborgCCA,
ParkhomenkoCCA, SAR.
"""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo._base import BaseModel
from cca_zoo.sparse import (
    ADMMCCA,
    PMDCCA,
    SAR,
    ParkhomenkoCCA,
    SpanCCA,
    WaijenborgCCA,
)

# ---------------------------------------------------------------------------
# Sparsity verification
# ---------------------------------------------------------------------------


def test_pmd_invariant_to_input_scale(two_views: list[np.ndarray]) -> None:
    """PMDCCA's fitted weights (up to sign) must not depend on input scale.

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
    model_a = PMDCCA(n_components=1, tau=0.5, max_iter=200, random_state=0).fit(
        two_views
    )
    model_b = PMDCCA(n_components=1, tau=0.5, max_iter=200, random_state=0).fit(
        scaled_views
    )
    for w_a, w_b in zip(model_a.weights_, model_b.weights_):
        # sign of the leading direction is arbitrary; align before comparing
        sign = np.sign((w_a * w_b).sum()) or 1.0
        np.testing.assert_allclose(w_a, sign * w_b, atol=1e-6)


def test_bisect_threshold_matches_a_from_scratch_bisection() -> None:
    """`_bisect_threshold`'s brentq solve matches an independent fixed bisection.

    `_bisect_threshold` used to run a hand-rolled, unconditional 50-iteration
    bisection with no early stop; replaced with `scipy.optimize.brentq` for
    the same monotonic root-find (3.65x faster across 500 random trials in a
    direct benchmark, 1.8x faster end-to-end in `PMDCCA.fit`). Pins the
    result against a from-scratch fixed-count bisection, independent of the
    function under test, across a range of vector sizes and scales.
    """
    from cca_zoo._utils._linalg import soft_threshold
    from cca_zoo.sparse._iterative import _bisect_threshold

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


def test_pmd_tau_controls_sparsity_monotonically(
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
        model = PMDCCA(n_components=1, tau=tau, max_iter=200, random_state=0).fit(
            two_views
        )
        nnz_by_tau.append(sum(int(np.sum(np.abs(w) > 1e-10)) for w in model.weights_))
    assert nnz_by_tau == sorted(nnz_by_tau), (
        f"nnz should be non-decreasing in tau, got {dict(zip(taus, nnz_by_tau))}"
    )
    # tau=1 imposes no real constraint (L1 bound = sqrt(p), the max
    # possible for a unit vector), so it must not be sparse.
    assert nnz_by_tau[-1] == sum(v.shape[1] for v in two_views)


def test_admm_stable_at_a_realistic_sample_size() -> None:
    """ADMMCCA's weights stay finite at n=200, not just the tiny n=50 examples.

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
    model = ADMMCCA(n_components=2, tau=0.3, mu=1.0, max_iter=500, random_state=0).fit(
        [X1, X2]
    )
    for w in model.weights_:
        assert np.all(np.isfinite(w))


def test_admm_block_satisfies_kkt_conditions() -> None:
    """ADMMCCA's inner linearised-ADMM block solve reaches a genuine KKT point.

    Regression test for two successive wrong objectives: the class originally
    (and, after a first "fix", still) solved a reduced-rank-regression-style
    loss `||Xw - target||^2` with the ball constraint on `w` itself. Reading
    the actual paper (Suo, Mineiro & Anandkumar 2017, Section 2.2) showed the
    real problem is linear-plus-L1 in `w`, constrained on the *score* `Xw`,
    not `w`:

        maximize_w  w^T X^T target - tau*||w||_1  s.t. ||Xw||_2 <= 1

    No off-the-shelf solver reliably handles this (scipy's `trust-constr`
    gets stuck at the L1 kink at the origin regardless of starting point, and
    a from-scratch subgradient method needs its own ball-constraint
    projection), so this test verifies the KKT conditions directly instead:
    at a constrained optimum, the dual variable recovered from any two active
    (nonzero) coordinates must agree, and every zeroed coordinate's
    subgradient residual must lie in [-tau, tau].
    """
    rng = np.random.default_rng(1)
    n, p = 60, 15
    X = rng.standard_normal((n, p))
    other_score = rng.standard_normal(n) * 0.5

    tau = 0.2
    model = ADMMCCA(
        n_components=1,
        tau=tau,
        mu=1.0,
        max_iter=1,
        admm_iter=20_000,
        tol=1e-14,
        random_state=0,
    )
    # A second, single-column "view" equal to other_score itself (weight
    # fixed at 1) makes `_fit_single`'s internal `s_other` for view 0 exactly
    # `other_score`, unnormalised -- with max_iter=1, view 0's block is
    # solved once against this fixed target before view 1 is ever touched.
    w = [np.zeros(p), np.array([1.0])]
    views = [X, other_score.reshape(-1, 1)]
    model._fit_single(views, w, 0)
    w_fit = w[0]

    Xw = X @ w_fit
    nrm = np.linalg.norm(Xw)
    assert nrm > 0.99, "constraint should be active for this tau"
    active = np.abs(w_fit) > 1e-6
    c = X.T @ other_score
    grad_term = X.T @ Xw / nrm

    lambdas = (c[active] - tau * np.sign(w_fit[active])) / grad_term[active]
    assert lambdas.min() > 0, "recovered dual variable must be non-negative"
    np.testing.assert_allclose(lambdas, lambdas.mean(), rtol=1e-3)

    zero_resid = c[~active] - lambdas.mean() * grad_term[~active]
    assert np.all(np.abs(zero_resid) <= tau + 1e-3)


def test_sar_finds_zero_weights_when_no_signal(two_views: list[np.ndarray]) -> None:
    """SAR's BIC selection should prefer the all-zero fit on pure noise.

    Where the true regression coefficient really is zero -- unlike
    every other class here, SAR has no user-set penalty strength to
    check sparsity against, so this checks the BIC selection itself
    rather than a fixed hyperparameter's effect.
    """
    model = SAR(n_components=1, max_iter=50, random_state=0).fit(two_views)
    for w in model.weights_:
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
    model = SAR(n_components=1, random_state=0).fit([x, y])
    for w in model.weights_:
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
    model = SAR(n_components=2, random_state=0).fit([x, y])
    zx, zy = model.transform([x, y])
    for d in range(2):
        assert np.corrcoef(zx[:, d], zy[:, d])[0, 1] > 0.9
    assert abs(np.corrcoef(zx[:, 0], zx[:, 1])[0, 1]) < 0.3


@pytest.mark.parametrize(
    "model",
    [
        PMDCCA(tau=0.3),
        ParkhomenkoCCA(tau=2.0),
        SpanCCA(span=3),
        ADMMCCA(tau=1.0),
        WaijenborgCCA(alpha=0.1, l1_ratio=1.0),
    ],
    ids=lambda m: type(m).__name__,
)
def test_penalty_zeroes_weights(
    model: BaseModel, correlated_views: list[np.ndarray]
) -> None:
    """Each sparsity penalty zeroes some, but not all, weights of a view."""
    model.set_params(random_state=0).fit(correlated_views)
    zero = [np.isclose(w, 0.0).mean() for w in model.weights_]
    assert any(0.0 < z < 1.0 for z in zero)
