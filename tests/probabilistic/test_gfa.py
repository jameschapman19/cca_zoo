"""Tests for GFA (Group Factor Analysis).

Unlike ProbabilisticCCA/VariationalBayesCCA, GFA has no dependency beyond
numpy/scikit-learn (closed-form coordinate-ascent VB, no numpyro/jax), so
these tests are not gated behind an importorskip and are not marked slow.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np

from cca_zoo.probabilistic import GFA


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
        "GFA(n_components=1, max_iter=10, random_state=0).fit(\n"
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
# ARD / dropK behaviour (the actual point of this class)
# ---------------------------------------------------------------------------


def test_gfa_drop_k_prunes_spurious_dimensions() -> None:
    """With more n_components than true shared factors, dropK removes some.

    VB coordinate ascent on an ARD model converges to *a* local optimum,
    not necessarily one matching the true generative dimensionality exactly
    (finite-sample noise can genuinely support an extra component from a
    given initialization) -- so this checks that pruning actually happens
    (fewer components than the requested upper bound), not that it recovers
    the "true" count of 2 exactly.
    """
    views = _make_low_rank_views(n=150, true_k=2, seed=0)
    model = GFA(n_components=4, drop_k=True, random_state=0).fit(views)
    assert model.n_components_ < 4
    for w in model.weights_:
        assert w.shape[1] == model.n_components_
    reconstructions = model.inverse_transform(model.transform(views))
    assert [r.shape for r in reconstructions] == [v.shape for v in views]


def test_gfa_drop_k_false_keeps_all_dimensions() -> None:
    """drop_k=False never prunes, even with clearly spurious dimensions."""
    views = _make_low_rank_views(n=150, true_k=2, seed=0)
    model = GFA(n_components=4, drop_k=False, max_iter=500, random_state=0).fit(views)
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

    model = GFA(n_components=4, random_state=0).fit([y1, y2])
    relevance = model.view_relevance_  # (n_views, n_components_)

    # At least one component should be private to view 0: small alpha in
    # view 0, huge (effectively zero-loading) alpha in view 1.
    private_candidates = relevance[1] / relevance[0]
    assert np.max(private_candidates) > 1e4, (
        f"Expected a component private to view 0, got relevance ratios "
        f"{private_candidates}"
    )
