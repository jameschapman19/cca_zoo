"""ManifoldCCA: CCA between graph-smooth embeddings of each view."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from sklearn.manifold import LocallyLinearEmbedding

from cca_zoo.linear import MCCA
from cca_zoo.nonparametric import ManifoldCCA
from cca_zoo.nonparametric._manifold_cca import _laplacian_operator


def _principal_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.svd(np.linalg.qr(a)[0].T @ np.linalg.qr(b)[0], compute_uv=False)


def test_duplicated_views_give_spectral_embedding() -> None:
    """Two copies of one view recover that view's Laplacian eigenmap."""
    x = np.random.default_rng(0).standard_normal((120, 6))
    model = ManifoldCCA(n_neighbors=10, n_components=3).fit([x, x.copy()])
    laplacian = _laplacian_operator(x - x.mean(0), 10, "nearest_neighbors", None)
    eigenmap = eigh(laplacian)[1][:, 1:4]
    for z in model.embedding_:
        assert np.all(_principal_cosines(z, eigenmap) > 1 - 1e-3)


def test_duplicated_views_extend_as_lle() -> None:
    """With method='lle', new points are placed as sklearn's LLE places them."""
    rng = np.random.default_rng(0)
    x, new = rng.standard_normal((100, 6)), rng.standard_normal((10, 6))
    model = ManifoldCCA(method="lle", n_neighbors=10, n_components=3).fit([x, x.copy()])
    lle = LocallyLinearEmbedding(n_neighbors=10, n_components=3).fit(x - x.mean(0))
    expected = lle.transform(new - x.mean(0))
    for z in model.transform([new, new.copy()]):
        assert np.all(_principal_cosines(z, expected) > 1 - 1e-2)


def test_transform_of_training_data_follows_the_embedding(
    two_views_small: list[np.ndarray],
) -> None:
    """Out-of-sample extension at the training points tracks their embedding."""
    for method in ("laplacian", "lle"):
        model = ManifoldCCA(method=method, n_neighbors=8).fit(two_views_small)
        for z, t in zip(model.embedding_, model.transform(two_views_small)):
            assert abs(np.corrcoef(z[:, 0], t[:, 0])[0, 1]) > 0.8


def test_relates_two_differently_wound_spirals() -> None:
    """A shared coordinate embedded as two different spirals defeats linear CCA."""
    rng = np.random.default_rng(0)

    def spirals(t: np.ndarray) -> list[np.ndarray]:
        return [
            np.column_stack([t * np.cos(f * t + p), t * np.sin(f * t + p)])
            + 0.03 * rng.standard_normal((t.size, 2))
            for f, p in ((1, 0), (2, 1))
        ]

    train, test = (spirals(np.sort(rng.uniform(0.5, 4 * np.pi, 200))) for _ in range(2))
    manifold = ManifoldCCA(n_neighbors=10).fit(train).score(test)
    assert manifold > MCCA(c=0.1, pca=False).fit(train).score(test) + 0.3
