import numpy as np

from cca_zoo.utils.cross_correlation import cross_corrcoef, cross_cov


def test_crosscorrcoef():
    X = np.random.rand(100, 5)
    Y = np.random.rand(100, 5) / 10

    m = np.corrcoef(X, Y, rowvar=False)[:5, 5:]
    n = cross_corrcoef(X, Y, rowvar=False)

    assert np.allclose(m, n)

def test_crosscov(bias=False):
    X = np.random.rand(100, 5)
    Y = np.random.rand(100, 5) / 10

    m = np.cov(X, Y, rowvar=False)[:5, 5:]
    n = cross_cov(X, Y, rowvar=False)

    assert np.allclose(m, n)
