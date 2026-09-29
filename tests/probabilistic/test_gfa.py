"""GFA: Bayesian CCA with an ARD prior per view, ported from R's CCAGFA."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
from sklearn.decomposition import PCA

from cca_zoo.probabilistic import GFA
from tests._helpers import principal_cosines


def _views(n: int = 150, shared: int = 2, private: int = 0) -> list[np.ndarray]:
    """Two views of ``shared`` common factors, plus ``private`` in view 1 only."""
    rng = np.random.default_rng(1)
    z = rng.standard_normal((n, shared))
    h = rng.standard_normal((n, private))
    return [
        z @ rng.standard_normal((shared, 5))
        + h @ rng.standard_normal((private, 5))
        + 0.1 * rng.standard_normal((n, 5)),
        z @ rng.standard_normal((shared, 4)) + 0.1 * rng.standard_normal((n, 4)),
    ]


def test_equal_noise_views_give_ppca_of_their_concatenation() -> None:
    """With equal noise in each view, GFA is PPCA of the stacked views.

    Probabilistic PCA's latent space is the top principal components.
    """
    views = _views(n=500)
    latent = GFA(2, random_state=0).fit(views).posterior_mean(views)
    pcs = PCA(2).fit_transform(np.hstack(views))
    assert np.all(principal_cosines(latent, pcs) > 1 - 1e-4)


@pytest.mark.parametrize(("drop_k", "kept"), [(True, range(1, 4)), (False, [4])])
def test_drop_k_prunes_unsupported_dimensions(drop_k: bool, kept: range) -> None:
    """With four dimensions for two factors, drop_k prunes some; without it, none."""
    model = GFA(n_components=4, drop_k=drop_k, max_iter=500, random_state=0)
    assert model.fit(_views()).n_components_ in kept


def test_ard_finds_a_factor_private_to_one_view() -> None:
    """A factor in view 1 only has a near-infinite ARD precision in view 2."""
    relevance = (
        GFA(n_components=4, random_state=0)
        .fit(_views(shared=1, private=1))
        .ard_precision_
    )
    assert np.max(relevance[1] / relevance[0]) > 1e4


def test_needs_neither_numpyro_nor_jax() -> None:
    """GFA runs with the base install, unlike the other probabilistic models."""
    script = (
        "import sys, numpy as np\n"
        "from cca_zoo.probabilistic import GFA\n"
        "GFA(max_iter=10).fit([np.random.rand(10, 3), np.random.rand(10, 3)])\n"
        "assert not {'numpyro', 'jax'} & set(sys.modules)\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_posterior_mean_needs_a_view_per_slot() -> None:
    """posterior_mean takes one entry per view, at least one observed."""
    model = GFA(max_iter=10).fit(_views())
    with pytest.raises(ValueError, match="Expected 2 views"):
        model.posterior_mean([None])
    with pytest.raises(ValueError, match="At least one view"):
        model.posterior_mean([None, None])
