"""ManifoldCCA: CCA between graph-smooth embeddings of each view."""

from __future__ import annotations

import numpy as np
from sklearn.manifold import LocallyLinearEmbedding, SpectralEmbedding

from cca_zoo.linear import MCCA
from cca_zoo.nonparametric import ManifoldCCA
from tests._helpers import principal_cosines


def test_duplicated_views_give_spectral_embedding() -> None:
    """Two copies of one view recover sklearn's Laplacian eigenmap of that view.

    SpectralEmbedding returns the normalised Laplacian's eigenvectors scaled by
    D^{-1/2}; rescaling by the square-root degree recovers them.
    """
    x = np.random.default_rng(0).standard_normal((120, 6))
    model = ManifoldCCA(n_neighbors=10, n_components=3).fit([x, x.copy()])
    se = SpectralEmbedding(n_components=3, n_neighbors=10, random_state=0)
    se.fit(x - x.mean(0))
    degree = np.asarray(se.affinity_matrix_.sum(axis=1)).ravel()
    eigenmap = np.sqrt(degree)[:, None] * se.embedding_
    for z in model.embedding_:
        assert np.all(principal_cosines(z, eigenmap) > 1 - 1e-3)


def test_duplicated_views_extend_as_lle() -> None:
    """With method='lle', new points are placed as sklearn's LLE places them."""
    rng = np.random.default_rng(0)
    x, new = rng.standard_normal((100, 6)), rng.standard_normal((10, 6))
    model = ManifoldCCA(method="lle", n_neighbors=10, n_components=3).fit([x, x.copy()])
    lle = LocallyLinearEmbedding(n_neighbors=10, n_components=3).fit(x - x.mean(0))
    expected = lle.transform(new - x.mean(0))
    for z in model.transform([new, new.copy()]):
        assert np.all(principal_cosines(z, expected) > 1 - 1e-2)


def test_transform_of_training_data_is_the_embedding(
    two_views_small: list[np.ndarray],
) -> None:
    """The out-of-sample extension returns each training point's embedding."""
    for kwargs in (
        {"method": "laplacian"},
        {"method": "laplacian", "affinity": "rbf", "n_operator_components": 8},
        {"method": "lle"},
    ):
        model = ManifoldCCA(n_neighbors=8, **kwargs).fit(two_views_small)
        for z, t in zip(model.embedding_, model.transform(two_views_small)):
            np.testing.assert_allclose(t, z, atol=1e-10)


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
    assert manifold > MCCA(shrinkage=0.1, pca=False).fit(train).score(test) + 0.3
