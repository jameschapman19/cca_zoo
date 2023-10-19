import numpy as np
import scipy.sparse as sp
from scipy.stats import loguniform
from sklearn.utils.validation import check_random_state

from cca_zoo.datasets import JointData
from cca_zoo.linear import (
    CCA,
    GCCA,
    GRCCA,
    MCCA,
    MPLS,
    PLS,
    PRCCA,
    SCCA_IPLS,
    SCCA_PMD,
    # AltMaxVar,
    ElasticCCA,
    PartialCCA,
    SCCA_Parkhomenko,
    SCCA_Span,
    rCCA,
)
from cca_zoo.linear._dummy import DummyCCA
from cca_zoo.model_selection import GridSearchCV, RandomizedSearchCV
from cca_zoo.nonparametric import KCCA

n = 50
rng = check_random_state(0)
X = rng.rand(n, 10)
Y = rng.rand(n, 11)
Z = rng.rand(n, 12)
X_sp = sp.random(n, 10, density=0.5, random_state=rng)
Y_sp = sp.random(n, 11, density=0.5, random_state=rng)
# centre the data
X -= X.mean(axis=0)
Y -= Y.mean(axis=0)
Z -= Z.mean(axis=0)
X_sp -= X_sp.mean(axis=0)
Y_sp -= Y_sp.mean(axis=0)


def test_initialisation():
    dummyCCA = DummyCCA(latent_dimensions=3)
    dummyCCA.fit((X, Y))
    # check that scores are positive in each dimension:
    scores = dummyCCA.score((X, Y))

    # Check if all scores are positive
    assert np.all(scores > 0), "Not all scores are positive."


def test_regularized_methods():
    # Test that linear regularized methods match PLS solution when using maximum regularisation.
    latent_dims = 2
    c = 1
    rcca = rCCA(latent_dimensions=latent_dims, c=[c, c]).fit([X, Y])
    mcca = MCCA(latent_dimensions=latent_dims, c=[c, c]).fit([X, Y])
    pls = PLS(latent_dimensions=latent_dims).fit([X, Y])
    gcca = GCCA(latent_dimensions=latent_dims, c=[c, c]).fit([X, Y])
    kernel = KCCA(
        latent_dimensions=latent_dims, c=[c, c], kernel=["linear", "linear"]
    ).fit((X, Y))
    mpls = MPLS(latent_dimensions=latent_dims).fit([X, Y])
    corr_gcca = gcca.score((X, Y))
    corr_mcca = mcca.score((X, Y))
    corr_kernel = kernel.score((X, Y))
    corr_pls = pls.score((X, Y))
    corr_rcca = rcca.score((X, Y))
    corr_mpls = mpls.score((X, Y))
    # Check the correlations from each unregularized method are the same
    assert np.testing.assert_array_almost_equal(corr_pls, corr_mcca, decimal=1) is None
    assert (
        np.testing.assert_array_almost_equal(corr_pls, corr_kernel, decimal=1) is None
    )
    assert np.testing.assert_array_almost_equal(corr_pls, corr_rcca, decimal=1) is None
    assert np.testing.assert_array_almost_equal(corr_pls, corr_mpls, decimal=1) is None


def test_sparse_methods():
    tau1 = [0, 3, 0.5, 0.7]
    tau2 = [0.7]
    param_grid = {"tau": [tau1, tau2]}
    pmd_cv = GridSearchCV(SCCA_PMD(random_state=rng), param_grid=param_grid).fit([X, Y])
    assert (pmd_cv.best_estimator_.weights[0] == 0).sum() > 0
    assert (pmd_cv.best_estimator_.weights[1] == 0).sum() > 0
    alpha1 = loguniform(1e-2, 2e-2)
    alpha2 = loguniform(1e-2, 2e-2)
    param_grid = {"alpha": [alpha1, alpha2], "l1_ratio": [[0.9], [0.9]]}
    elastic_cv = RandomizedSearchCV(
        ElasticCCA(random_state=rng), param_distributions=param_grid, n_iter=4
    ).fit([X, Y])
    alpha1 = loguniform(1e-2, 2e-2)
    alpha2 = loguniform(1e-2, 2e-2)
    param_grid = {"alpha": [alpha1, alpha2]}
    scca_cv = RandomizedSearchCV(
        SCCA_IPLS(random_state=rng), param_distributions=param_grid
    ).fit([X, Y])
    tau1 = [2e-1]
    tau2 = [2e-1]
    param_grid = {"tau": [tau1, tau2]}
    parkhomenko_cv = GridSearchCV(
        SCCA_Parkhomenko(random_state=rng), param_grid=param_grid
    ).fit([X, Y])
    tau1 = [1e-1]
    tau2 = [1e-1]
    assert (scca_cv.best_estimator_.weights[0] == 0).sum() > 0
    assert (scca_cv.best_estimator_.weights[1] == 0).sum() > 0
    assert (parkhomenko_cv.best_estimator_.weights[0] == 0).sum() > 0
    assert (parkhomenko_cv.best_estimator_.weights[1] == 0).sum() > 0
    assert (elastic_cv.best_estimator_.weights[0] == 0).sum() > 0
    assert (elastic_cv.best_estimator_.weights[1] == 0).sum() > 0


def test_weighted_GCCA_methods():
    # TODO we have view weighted GCCALoss and missing observation GCCALoss
    latent_dims = 2
    c = 0
    unweighted_gcca = GCCA(latent_dimensions=latent_dims, c=[c, c]).fit([X, Y])
    deweighted_gcca = GCCA(
        latent_dimensions=latent_dims, c=[c, c], view_weights=[0.5, 0.5]
    ).fit([X, Y])
    corr_unweighted_gcca = unweighted_gcca.score((X, Y))
    corr_deweighted_gcca = deweighted_gcca.score((X, Y))
    # Check the correlations from each unregularized method are the same
    K = np.ones((2, X.shape[0]))
    K[0, 200:] = 0
    unobserved_gcca = GCCA(latent_dimensions=latent_dims, c=[c, c]).fit((X, Y), K=K)
    assert (
        np.testing.assert_array_almost_equal(
            corr_unweighted_gcca, corr_deweighted_gcca, decimal=1
        )
        is None
    )


def test_l0():
    span_cca = SCCA_Span(
        latent_dimensions=1, regularisation="l0", tau=[2, 2], random_state=rng
    ).fit([X, Y])
    assert (np.abs(span_cca.weights[0]) > 1e-5).sum() == 2
    assert (np.abs(span_cca.weights[1]) > 1e-5).sum() == 2


def test_partialcca():
    # Tests that partial CCALoss scores are not correlated with partials
    pcca = PartialCCA(latent_dimensions=3)
    pcca.fit((X, Y), partials=Z)
    assert np.allclose(
        np.corrcoef(pcca.transform((X, Y), partials=Z)[0], Z, rowvar=False)[:3, :3]
        - np.eye(3),
        0,
        atol=0.001,
    )


def test_PRCCA():
    # Test that PRCCA works
    prcca = PRCCA(latent_dimensions=2, c=[0, 0]).fit(
        (X, Y), idxs=(np.arange(10), np.arange(11))
    )
    cca = CCA(latent_dimensions=2).fit([X, Y])
    assert (
        np.testing.assert_array_almost_equal(
            cca.score((X, Y)), prcca.score((X, Y)), decimal=1
        )
        is None
    )
    prcca = PRCCA(latent_dimensions=2, c=[1, 1]).fit(
        (X, Y), idxs=(np.arange(10), np.arange(11))
    )
    pls = PLS(latent_dimensions=2).fit([X, Y])
    assert (
        np.testing.assert_array_almost_equal(
            pls.score((X, Y)), prcca.score((X, Y)), decimal=1
        )
        is None
    )


def test_GRCCA():
    feature_group_1 = np.zeros(X.shape[1], dtype=int)
    feature_group_1[:3] = 1
    feature_group_1[3:6] = 2
    feature_group_2 = np.zeros(Y.shape[1], dtype=int)
    feature_group_2[:3] = 1
    feature_group_2[3:6] = 2
    feature_group_3 = np.zeros(Z.shape[1], dtype=int)
    feature_group_3[:3] = 1
    feature_group_3[3:6] = 2
    # Test that GRCCA works
    grcca = GRCCA(latent_dimensions=1, c=[100, 0], mu=0).fit(
        (X, Y), feature_groups=[feature_group_1, feature_group_2]
    )
    grcca.score((X, Y))
    grcca.transform((X, Y))
    grcca = GRCCA(c=[100, 0, 50]).fit(
        (X, Y, Z), feature_groups=[feature_group_1, feature_group_2, feature_group_3]
    )


def test_deflation():
    # test deflation works with pls and cca using SCCA_PMD
    ccamodel = SCCA_PMD(latent_dimensions=2, tau=0.9, random_state=rng)
    ccamodel.fit([X, Y])
    pls = PLS(latent_dimensions=2)
    pls.fit([X, Y])
    plsmodel = SCCA_PMD(latent_dimensions=2, tau=0.9, random_state=rng)
    plsmodel.fit([X, Y])
    assert (
        np.testing.assert_array_almost_equal(
            np.abs(pls.score((X, Y))), np.abs(plsmodel.score((X, Y))), decimal=1
        )
        is None
    )
