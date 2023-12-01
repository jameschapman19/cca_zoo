import numpy as np

from cca_zoo._utils._cross_correlation import cross_corrcoef, cross_cov

N = 50
features = [4, 6]


def test_crosscorrcoef():
    X = np.random.rand(N, features[0])
    Y = np.random.rand(N, features[1]) / 10

    m = np.corrcoef(X, Y, rowvar=False)[:4, 4:]
    n = cross_corrcoef(X, Y, rowvar=False)

    assert np.allclose(m, n)


def test_crosscov(bias=False):
    X = np.random.rand(N, features[0])
    Y = np.random.rand(N, features[1]) / 10

    m = np.cov(X, Y, rowvar=False)[:4, 4:]
    n = cross_cov(X, Y, rowvar=False)

    assert np.allclose(m, n)
