"""GaussianProcessCCA: a Gaussian process encoder per view."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.gaussian_process.kernels import RBF, DotProduct

from cca_zoo.gam import GAMCCA
from cca_zoo.gp import GaussianProcessCCA


def test_captures_an_interaction_additive_models_miss() -> None:
    """A joint kernel relates u * v to (u, v); an additive encoder cannot."""
    rng = np.random.default_rng(0)
    u, v = rng.standard_normal((2, 600))
    views = [
        np.column_stack([u, v]) + 0.2 * rng.standard_normal((600, 2)),
        np.column_stack([u * v, u * v]) + 0.2 * rng.standard_normal((600, 2)),
    ]
    train, test = [x[:300] for x in views], [x[300:] for x in views]
    gp = GaussianProcessCCA(random_state=0).fit(train).score(test)
    assert gp > 0.7
    assert gp > GAMCCA().fit(train).score(test) + 0.05


@pytest.mark.parametrize("n_inducing", [None, 15])
def test_return_std_gives_positive_uncertainty(
    two_views_small: list[np.ndarray], n_inducing: int | None
) -> None:
    """Exact and sparse posteriors give a positive std for every score."""
    model = GaussianProcessCCA(n_components=2, n_inducing=n_inducing, random_state=0)
    means, stds = model.fit(two_views_small).transform(two_views_small, return_std=True)
    for mean, std in zip(means, stds):
        assert std.shape == mean.shape
        assert np.all(std > 0)


def test_inducing_points_beyond_the_data_is_exact(
    two_views_small: list[np.ndarray],
) -> None:
    """More inducing points than samples uses every sample: exact inference."""
    model = GaussianProcessCCA(n_inducing=1000, random_state=0).fit(two_views_small)
    assert all(enc.inducing_.shape[0] == 30 for enc in model.encoders_)


def test_kernel_per_view(two_views_small: list[np.ndarray]) -> None:
    """A list of kernels gives each view its own."""
    model = GaussianProcessCCA(kernel=[DotProduct(), RBF()], random_state=0)
    kernels = [enc.kernel_ for enc in model.fit(two_views_small).encoders_]
    assert isinstance(kernels[0], DotProduct) and isinstance(kernels[1], RBF)
