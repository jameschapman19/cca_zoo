"""Tests for GaussianProcessCCA.

Like GAMCCA, GaussianProcessCCA has no optional dependency (it is built entirely on
scikit-learn's GaussianProcessRegressor, already required by cca_zoo), so
these tests run unconditionally.
"""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo.gp import GaussianProcessCCA
from cca_zoo.gp._gpcca import _GpEncoder


def _make_model(n_components: int = 1, **kwargs: object) -> GaussianProcessCCA:
    kwargs.setdefault("random_state", 0)
    return GaussianProcessCCA(n_components=n_components, **kwargs)


# ---------------------------------------------------------------------------
# return_std uncertainty output
# ---------------------------------------------------------------------------


def test_transform_return_std_shapes_and_positive(
    two_views_small: list[np.ndarray],
) -> None:
    """transform(..., return_std=True) returns matching-shape, positive stds."""
    k = 2
    model = _make_model(n_components=k).fit(two_views_small)
    means, stds = model.transform(two_views_small, return_std=True)
    n = two_views_small[0].shape[0]
    assert len(means) == 2
    assert len(stds) == 2
    for mean, std in zip(means, stds):
        assert mean.shape == (n, k)
        assert std.shape == (n, k)
        assert np.all(std > 0)


# ---------------------------------------------------------------------------
# Per-view parameters
# ---------------------------------------------------------------------------


def test_per_view_kernel_list(two_views_small: list[np.ndarray]) -> None:
    """A per-view kernel list uses a different kernel per view."""
    from sklearn.gaussian_process.kernels import RBF, DotProduct

    model = _make_model(kernel=[DotProduct(), RBF()]).fit(two_views_small)
    assert isinstance(model.encoders_[0].kernel_, DotProduct)
    assert isinstance(model.encoders_[1].kernel_, RBF)


# ---------------------------------------------------------------------------
# sparse (inducing-point) approximation
# ---------------------------------------------------------------------------


def test_sparse_return_std_shapes_and_positive(
    two_views_small: list[np.ndarray],
) -> None:
    """Sparse transform(..., return_std=True) returns positive, matching-shape stds."""
    k = 2
    n = two_views_small[0].shape[0]
    model = _make_model(n_components=k, n_inducing=n // 2).fit(two_views_small)
    means, stds = model.transform(two_views_small, return_std=True)
    for mean, std in zip(means, stds):
        assert mean.shape == (n, k)
        assert std.shape == (n, k)
        assert np.all(std > 0)


def test_n_inducing_at_least_n_samples_falls_back_to_exact(
    two_views_small: list[np.ndarray],
) -> None:
    """n_inducing >= n_samples is equivalent to exact (dense) GP inference."""
    n = two_views_small[0].shape[0]
    model = _make_model(n_inducing=10 * n).fit(two_views_small)
    for enc in model.encoders_:
        assert isinstance(enc, _GpEncoder)
        assert enc.inducing_.shape[0] == n


@pytest.mark.slow
def test_sparse_scales_to_large_sample_sizes() -> None:
    """Sparse GaussianProcessCCA fits at a size that would defeat exact GP inference.

    Exact GP inference redoes an O(n^3) Cholesky factorisation at every
    Newton step of every inner/outer round, which would be impractically
    slow here; the sparse approximation should still recover the
    underlying correlation.
    """
    rng = np.random.default_rng(0)
    n_train, n_test, noise = 4000, 500, 0.3
    n = n_train + n_test
    z = rng.standard_normal(n)
    X1 = np.column_stack([z + noise * rng.standard_normal(n) for _ in range(3)])
    X2 = np.column_stack([z**2 + noise * rng.standard_normal(n) for _ in range(3)])
    X1_tr, X1_te = X1[:n_train], X1[n_train:]
    X2_tr, X2_te = X2[:n_train], X2[n_train:]

    model = GaussianProcessCCA(n_components=1, random_state=0, n_inducing=100)
    test_corr = model.fit([X1_tr, X2_tr]).score([X1_te, X2_te])
    assert test_corr > 0.7, (
        f"Expected substantial held-out correlation, got {test_corr}"
    )


@pytest.mark.slow
def test_gpcca_outperforms_others_on_genuine_interaction() -> None:
    """GaussianProcessCCA beats GAMCCA, TreeCCA and RidgeCCA on an interaction task.

    View 1 is two noisy independent factors ``u, v``; view 2 is a noisy
    copy of their *interaction* ``u * v`` -- not additively separable into
    a function of ``u`` plus a function of ``v``. GAMCCA's additive-spline
    encoder structurally cannot represent this; ``TreeCCA`` can only
    approximate it via multivariate splits, and at its default
    ``colsample_bytree`` a single tree is often starved of joint access to
    both features. GaussianProcessCCA's joint (non-additive) RBF kernel
    represents the interaction directly.

    Marked slow since it also requires TreeCCA's optional ``xgboost``
    dependency, not part of the base ``dev`` install.
    """
    pytest.importorskip("xgboost", reason="xgboost is not installed")
    from cca_zoo.gam import GAMCCA
    from cca_zoo.linear import RidgeCCA
    from cca_zoo.tree import XGBoostCCA

    rng = np.random.default_rng(0)
    n_train, n_test, noise = 300, 300, 0.2
    n = n_train + n_test
    u = rng.standard_normal(n)
    v = rng.standard_normal(n)
    interaction = u * v
    X1 = np.column_stack([u, v]) + noise * rng.standard_normal((n, 2))
    X2 = np.column_stack([interaction, interaction]) + noise * rng.standard_normal(
        (n, 2)
    )
    X1_tr, X1_te = X1[:n_train], X1[n_train:]
    X2_tr, X2_te = X2[:n_train], X2[n_train:]

    gp = GaussianProcessCCA(n_components=1, random_state=0)
    gp_test = gp.fit([X1_tr, X2_tr]).score([X1_te, X2_te])

    gam = GAMCCA(n_components=1)
    gam_test = gam.fit([X1_tr, X2_tr]).score([X1_te, X2_te])

    tree = XGBoostCCA(n_components=1, n_estimators=150, max_depth=5, random_state=0)
    tree_test = tree.fit([X1_tr, X2_tr]).score([X1_te, X2_te])

    rcca = RidgeCCA(n_components=1, c=[0.3, 0.3])
    rcca_test = rcca.fit([X1_tr, X2_tr]).score([X1_te, X2_te])

    assert gp_test > 0.7, (
        f"Expected GaussianProcessCCA to recover the interaction, got {gp_test}"
    )
    assert gp_test > gam_test + 0.05, (
        f"Expected GaussianProcessCCA ({gp_test}) to beat additive GAMCCA ({gam_test}) "
        f"on a genuine feature interaction"
    )
    assert gp_test > tree_test + 0.05, (
        f"Expected GaussianProcessCCA ({gp_test}) to beat TreeCCA ({tree_test}) at "
        f"TreeCCA's default colsample_bytree"
    )
    assert gp_test > rcca_test + 0.3, (
        f"Expected GaussianProcessCCA ({gp_test}) to clearly beat linear "
        f"RidgeCCA ({rcca_test})"
    )
