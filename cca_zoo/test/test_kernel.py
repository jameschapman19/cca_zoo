"""
Test the kernel methods
"""


import numpy as np
from cca_zoo.models import KCCA, KTCCA, KGCCA, MCCA, GCCA, TCCA

np.random.seed(1)
X = np.random.normal(size=(100, 10))
Y = np.random.normal(size=(100, 10))
Z = np.random.normal(size=(100, 10))


# test that MCCA is the same as KCCA with linear kernel
def test_MCCA_KCCA():
    kcca = KCCA(latent_dims=2)
    mcca = MCCA(latent_dims=2)
    kcca.fit([X, Y, Z])
    mcca.fit([X, Y, Z])
    mcca_score = mcca.score([X, Y, Z])
    kcca_score = kcca.score([X, Y, Z])
    assert np.allclose(mcca_score, kcca_score)


# test that GCCA is the same as KGCCA with linear kernel
def test_GCCA_KGCCA():
    kgcca = KGCCA(latent_dims=2)
    gcca = GCCA(latent_dims=2)
    kgcca.fit([X, Y, Z])
    gcca.fit([X, Y, Z])
    gcca_score = gcca.score([X, Y, Z])
    kgcca_score = kgcca.score([X, Y, Z])
    assert np.allclose(gcca_score, kgcca_score)


# test that TCCA is the same as KTCCA with linear kernel
def test_TCCA_KTCCA():
    ktcca = KTCCA(latent_dims=2)
    tcca = TCCA(latent_dims=2)
    ktcca.fit([X, Y, Z])
    tcca.fit([X, Y, Z])
    tcca_score = tcca.score([X, Y, Z])
    ktcca_score = ktcca.score([X, Y, Z])
    assert np.allclose(tcca_score, ktcca_score)


def test_rbf_kernel():
    kcca = KCCA(latent_dims=2, kernel="rbf").fit((X, Y, Z))
    kgcca = KGCCA(latent_dims=2, kernel="rbf").fit((X, Y, Z))
    ktcca = KTCCA(latent_dims=2, kernel="rbf").fit((X, Y, Z))


def test_poly_kernel():
    kcca = KCCA(latent_dims=2, kernel="poly").fit((X, Y, Z))
    kgcca = KGCCA(latent_dims=2, kernel="poly").fit((X, Y, Z))
    ktcca = KTCCA(latent_dims=2, kernel="poly").fit((X, Y, Z))


def test_sigmoid_kernel():
    kcca = KCCA(latent_dims=2, kernel="sigmoid").fit((X, Y, Z))
    kgcca = KGCCA(latent_dims=2, kernel="sigmoid").fit((X, Y, Z))
    ktcca = KTCCA(latent_dims=2, kernel="sigmoid").fit((X, Y, Z))


def test_cosine_kernel():
    kcca = KCCA(latent_dims=2, kernel="cosine").fit((X, Y, Z))
    kgcca = KGCCA(latent_dims=2, kernel="cosine").fit((X, Y, Z))
    ktcca = KTCCA(latent_dims=2, kernel="cosine").fit((X, Y, Z))


def test_callable_kernel():
    def my_kernel(X, Y, **kwargs):
        return np.dot(X, Y.T)

    kcca = KCCA(latent_dims=2, kernel=my_kernel).fit((X, Y, Z))
    kgcca = KGCCA(latent_dims=2, kernel=my_kernel).fit((X, Y, Z))
    ktcca = KTCCA(latent_dims=2, kernel=my_kernel).fit((X, Y, Z))
