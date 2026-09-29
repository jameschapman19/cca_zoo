"""The kernel models: KCCA, KGCCA and KTCCA."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import pairwise_kernels
from sklearn.preprocessing import KernelCenterer

from cca_zoo.linear import GCCA, MCCA
from cca_zoo.nonparametric import KCCA, KGCCA, KTCCA


def _assert_same_scores(a: list[np.ndarray], b: list[np.ndarray]) -> None:
    """Equal scores up to the sign of each component."""
    for x, y in zip(a, b):
        np.testing.assert_allclose(np.abs(x), np.abs(y), atol=1e-8)


def _views(seed: int, n: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 2))
    return [
        np.c_[z, rng.standard_normal((n, 3))] + 5.0,
        np.c_[np.sin(z), rng.standard_normal((n, 2))] + 5.0,
    ]


@pytest.mark.parametrize("shrinkage", [0.01, 0.5, 1.0])
@pytest.mark.parametrize(("kernel_cls", "linear_cls"), [(KCCA, MCCA), (KGCCA, GCCA)])
def test_linear_kernel_is_the_linear_model(
    kernel_cls: type, linear_cls: type, shrinkage: float
) -> None:
    """With a linear kernel, a kernel model is its linear counterpart."""
    train, test = _views(0, 50), _views(1, 20)
    kernel = kernel_cls(n_components=2, shrinkage=shrinkage).fit(train)
    linear = linear_cls(n_components=2, shrinkage=shrinkage).fit(train)
    _assert_same_scores(kernel.transform(test), linear.transform(test))


@pytest.mark.parametrize("shrinkage", [0.01, 0.5, 1.0])
def test_kcca_is_mcca_on_the_exact_nystroem_features(shrinkage: float) -> None:
    """KCCA is MCCA on Nystroem features using every training row as a landmark."""
    train, test = _views(0, 50), _views(1, 20)
    nystroem = [
        Nystroem(kernel="rbf", gamma=0.2, n_components=50).fit(v) for v in train
    ]

    def features(views: list[np.ndarray]) -> list[np.ndarray]:
        return [m.transform(v) for m, v in zip(nystroem, views)]

    kcca = KCCA(
        n_components=2, center=False, kernel="rbf", gamma=0.2, shrinkage=shrinkage
    ).fit(train)
    mcca = MCCA(n_components=2, shrinkage=shrinkage).fit(features(train))
    _assert_same_scores(kcca.transform(test), mcca.transform(features(test)))


@pytest.mark.parametrize("cls", [KCCA, KGCCA, KTCCA])
def test_scores_are_centred_kernel_expansions(cls: type) -> None:
    """Scores are the training-centred kernel of the new rows times weights_."""
    train, test = _views(0, 40), _views(1, 10)
    model = cls(kernel="rbf").fit(train)
    for i, scores in enumerate(model.transform(test)):
        centerer = KernelCenterer().fit(
            pairwise_kernels(model.views_fit_[i], metric="rbf")
        )
        kernel = pairwise_kernels(
            test[i] - model.means_[i], model.views_fit_[i], metric="rbf"
        )
        np.testing.assert_allclose(
            scores, centerer.transform(kernel) @ model.weights_[i], atol=1e-10
        )


@pytest.mark.parametrize("cls", [KCCA, KGCCA, KTCCA])
def test_training_scores_are_centred(cls: type) -> None:
    """Training scores have zero mean, as for the linear models."""
    train = _views(0, 40)
    for scores in cls(n_components=2, kernel="rbf").fit(train).transform(train):
        np.testing.assert_allclose(scores.mean(axis=0), 0.0, atol=1e-10)
