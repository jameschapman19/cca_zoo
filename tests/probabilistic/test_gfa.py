"""Tests for GFA (Group Factor Analysis).

Unlike ProbabilisticCCA/VariationalBayesCCA, GFA has no dependency beyond
numpy/scikit-learn (closed-form coordinate-ascent VB, no numpyro/jax), so
these tests are not gated behind an importorskip and are not marked slow.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from cca_zoo.probabilistic import GFA


def _make_small_views(
    n: int = 20, p1: int = 4, p2: int = 4, seed: int = 0
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal((n, p1))
    x2 = rng.standard_normal((n, p2))
    return [x1, x2]


def _make_low_rank_views(
    n: int = 100, true_k: int = 2, p1: int = 6, p2: int = 5, seed: int = 0
) -> list[np.ndarray]:
    """Two views sharing exactly ``true_k`` latent dimensions, plus noise."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, true_k))
    x1 = z @ rng.standard_normal((true_k, p1)) + 0.1 * rng.standard_normal((n, p1))
    x2 = z @ rng.standard_normal((true_k, p2)) + 0.1 * rng.standard_normal((n, p2))
    return [x1, x2]


# ---------------------------------------------------------------------------
# fit completes / basic attributes
# ---------------------------------------------------------------------------


def test_gfa_fit_completes() -> None:
    """GFA.fit completes on two-view data without error."""
    views = _make_small_views()
    model = GFA(latent_dimensions=2, max_iter=200, random_state=0)
    fitted = model.fit(views)
    assert fitted is model


def test_gfa_fit_sets_params() -> None:
    """GFA.fit stores posterior samples, n_components_, and view_relevance_."""
    views = _make_small_views()
    model = GFA(latent_dimensions=2, max_iter=200, random_state=0).fit(views)
    assert hasattr(model, "posterior_samples_")
    assert hasattr(model, "n_components_")
    assert hasattr(model, "n_iter_")
    assert model.view_relevance_.shape == (2, model.n_components_)


def test_gfa_does_not_import_numpyro_or_jax() -> None:
    """GFA has no dependency beyond numpy/scikit-learn.

    Verified in a fresh subprocess so this can't be confused by
    numpyro/jax already being imported elsewhere in the test session.
    """
    script = (
        "import sys\n"
        "from cca_zoo.probabilistic import GFA\n"
        "import numpy as np\n"
        "rng = np.random.default_rng(0)\n"
        "GFA(latent_dimensions=1, max_iter=10).fit(\n"
        "    [rng.standard_normal((10, 3)), rng.standard_normal((10, 3))]\n"
        ")\n"
        "assert 'numpyro' not in sys.modules, sys.modules.keys()\n"
        "assert 'jax' not in sys.modules, sys.modules.keys()\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


# ---------------------------------------------------------------------------
# transform / scoring
# ---------------------------------------------------------------------------


def test_gfa_transform_output_shapes() -> None:
    """GFA.transform returns a single-element list of the right shape."""
    n, k = 20, 2
    views = _make_small_views(n=n)
    model = GFA(latent_dimensions=k, drop_k=False, max_iter=200, random_state=0).fit(
        views
    )
    result = model.transform(views)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].shape == (n, model.n_components_)


def test_gfa_score_is_finite() -> None:
    """score() must not be nan (regression for the shared single-z transform bug)."""
    views = _make_low_rank_views(n=60)
    model = GFA(latent_dimensions=2, drop_k=False, max_iter=200, random_state=0).fit(
        views
    )
    s = model.score(views)
    assert s.shape == (2,)
    assert np.all(np.isfinite(s))


def test_gfa_get_factor_loadings_one_per_view() -> None:
    """get_factor_loadings returns one array per view."""
    views = _make_low_rank_views(n=60)
    model = GFA(latent_dimensions=2, drop_k=False, max_iter=200, random_state=0).fit(
        views
    )
    loadings = model.get_factor_loadings(views)
    assert len(loadings) == 2
    for loading, view in zip(loadings, views):
        assert loading.shape == (view.shape[1], model.n_components_)


def test_gfa_log_likelihood_is_finite() -> None:
    """log_likelihood returns a finite scalar."""
    views = _make_low_rank_views(n=60)
    model = GFA(latent_dimensions=2, drop_k=False, max_iter=200, random_state=0).fit(
        views
    )
    ll = model.log_likelihood(views)
    assert isinstance(ll, float)
    assert np.isfinite(ll)


# ---------------------------------------------------------------------------
# latent_dimensions / view-count handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 3])
def test_gfa_latent_dimensions(k: int) -> None:
    """GFA can be instantiated with various latent_dimensions upper bounds."""
    views = _make_small_views(n=15)
    model = GFA(latent_dimensions=k, drop_k=False, max_iter=200, random_state=0)
    model.fit(views)
    assert model.latent_dimensions == k
    assert model.n_components_ == k  # drop_k=False: no pruning


def test_gfa_supports_more_than_two_views() -> None:
    """GFA's generative model is view-count generic (>=2)."""
    rng = np.random.default_rng(0)
    three_views = [rng.standard_normal((15, 4)) for _ in range(3)]
    model = GFA(latent_dimensions=1, max_iter=200, random_state=0)
    model.fit(three_views)
    assert len(model.weights_) == 3


def test_gfa_rejects_single_view() -> None:
    """GFA raises ValueError when given fewer than 2 views."""
    rng = np.random.default_rng(0)
    one_view = [rng.standard_normal((15, 4))]
    model = GFA(latent_dimensions=1, max_iter=200, random_state=0)
    with pytest.raises(ValueError, match="views"):
        model.fit(one_view)


# ---------------------------------------------------------------------------
# Reproducibility / center=False
# ---------------------------------------------------------------------------


def test_gfa_reproducibility() -> None:
    """Same random_state gives identical weights."""
    views = _make_small_views()
    kwargs = dict(latent_dimensions=2, max_iter=200, random_state=42)
    w1 = GFA(**kwargs).fit(views).weights
    w2 = GFA(**kwargs).fit(views).weights
    for a, b in zip(w1, w2):
        np.testing.assert_array_equal(a, b)


def test_gfa_center_false() -> None:
    """GFA works with center=False."""
    views = _make_small_views()
    model = GFA(latent_dimensions=1, center=False, max_iter=200, random_state=0)
    model.fit(views)
    result = model.transform(views)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# ARD / dropK behaviour (the actual point of this class)
# ---------------------------------------------------------------------------


def test_gfa_drop_k_prunes_spurious_dimensions() -> None:
    """With more latent_dimensions than true shared factors, dropK removes some.

    VB coordinate ascent on an ARD model converges to *a* local optimum,
    not necessarily one matching the true generative dimensionality exactly
    (finite-sample noise can genuinely support an extra component from a
    given initialization) -- so this checks that pruning actually happens
    (fewer components than the requested upper bound), not that it recovers
    the "true" count of 2 exactly.
    """
    views = _make_low_rank_views(n=150, true_k=2, seed=0)
    model = GFA(latent_dimensions=4, drop_k=True, random_state=0).fit(views)
    assert model.n_components_ < 4
    for w in model.weights_:
        assert w.shape[1] == model.n_components_


def test_gfa_drop_k_false_keeps_all_dimensions() -> None:
    """drop_k=False never prunes, even with clearly spurious dimensions."""
    views = _make_low_rank_views(n=150, true_k=2, seed=0)
    model = GFA(latent_dimensions=4, drop_k=False, max_iter=500, random_state=0).fit(
        views
    )
    assert model.n_components_ == 4


def test_gfa_identifies_private_factor() -> None:
    """A factor present in only one view gets a huge ARD precision in the other.

    This is the actual distinguishing mechanism of GFA vs.
    VariationalBayesCCA: per-(view, component) ARD lets a component be
    "private" to one view (tiny alpha there, but astronomically large --
    i.e. an ~zero loading -- in every other view) rather than forcing every
    retained component to be shared by construction.
    """
    rng = np.random.default_rng(1)
    n = 150
    z_shared = rng.standard_normal((n, 1))
    z_private = rng.standard_normal((n, 1))
    y1 = (
        z_shared @ rng.standard_normal((1, 5))
        + z_private @ rng.standard_normal((1, 5))
        + 0.1 * rng.standard_normal((n, 5))
    )
    y2 = z_shared @ rng.standard_normal((1, 4)) + 0.1 * rng.standard_normal((n, 4))

    model = GFA(latent_dimensions=4, random_state=0).fit([y1, y2])
    relevance = model.view_relevance_  # (n_views, n_components_)

    # At least one component should be private to view 0: small alpha in
    # view 0, huge (effectively zero-loading) alpha in view 1.
    private_candidates = relevance[1] / relevance[0]
    assert np.max(private_candidates) > 1e4, (
        f"Expected a component private to view 0, got relevance ratios "
        f"{private_candidates}"
    )
