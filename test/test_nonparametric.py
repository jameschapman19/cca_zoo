"""
Test the kernel methods
"""

import numpy as np
import pytest
from cca_zoo.linear import GCCA, MCCA, TCCA
from cca_zoo.nonparametric import KCCA, KGCCA, KTCCA, NCCA


@pytest.fixture
def data():
    N = 50
    features = [4, 6, 8]
    np.random.seed(1)
    X = np.random.normal(size=(N, features[0]))
    Y = np.random.normal(size=(N, features[1]))
    Z = np.random.normal(size=(N, features[2]))
    return X, Y, Z


def test_equivalence_with_linear_kernel(data):
    X, Y, Z = data
    kernel_tests = [(MCCA, KCCA), (GCCA, KGCCA), (TCCA, KTCCA)]

    for model1, model2 in kernel_tests:
        instance1 = model1(latent_dimensions=2).fit([X, Y, Z])
        instance2 = model2(latent_dimensions=2).fit([X, Y, Z])
        score1 = instance1.score([X, Y, Z])
        score2 = instance2.score([X, Y, Z])
        assert np.allclose(
            score1, score2, rtol=1e-2
        ), f"Scores differ for {model1} and {model2}"


@pytest.mark.parametrize("kernel", ["rbf", "poly", "sigmoid", "cosine"])
def test_kernel_types(kernel, data):
    X, Y, Z = data
    models = [KCCA, KGCCA, KTCCA]

    for model in models:
        instance = model(latent_dimensions=2, kernel=kernel).fit((X, Y, Z))
        assert instance is not None, f"Failed for model {model} with kernel {kernel}"


def test_callable_kernel(data):
    X, Y, Z = data

    def my_kernel(X, Y, **kwargs):
        return np.dot(X, Y.T)

    models = [KCCA, KGCCA, KTCCA]

    for model in models:
        instance = model(latent_dimensions=2, kernel=my_kernel).fit((X, Y, Z))
        assert instance is not None, f"Failed for model {model} with custom kernel"


def test_NCCA(data):
    X, Y = data[:2]
    ncca = NCCA(latent_dimensions=1).fit((X, Y))
    corr_ncca = ncca.score((X, Y))
    assert corr_ncca > 0.9, f"Expected score > 0.9 but got {corr_ncca}"
