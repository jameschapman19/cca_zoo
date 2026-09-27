"""The kernel models: KCCA, KGCCA and KTCCA."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import pairwise_kernels

from cca_zoo.linear import CCA
from cca_zoo.nonparametric import KCCA, KGCCA, KTCCA


def test_linear_kernel_is_cca(correlated_views: list[np.ndarray]) -> None:
    """With a linear kernel and little regularisation, KCCA is CCA."""
    np.testing.assert_allclose(
        KCCA(n_components=2, shrinkage=1e-4)
        .fit(correlated_views)
        .score(correlated_views),
        CCA(n_components=2).fit(correlated_views).score(correlated_views),
        atol=1e-3,
    )


@pytest.mark.parametrize("cls", [KCCA, KGCCA, KTCCA])
def test_scores_are_kernel_expansions(cls: type) -> None:
    """Scores are the kernel against the centred training data times weights_."""
    rng = np.random.default_rng(0)
    views = [5.0 + rng.standard_normal((40, 3)) for _ in range(2)]
    model = cls(kernel="rbf").fit(views)
    for i, scores in enumerate(model.transform(views)):
        kernel = pairwise_kernels(model.train_views_[i], metric="rbf")
        np.testing.assert_allclose(scores, kernel @ model.weights_[i], atol=1e-10)
