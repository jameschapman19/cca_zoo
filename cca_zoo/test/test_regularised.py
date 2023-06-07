import numpy as np
import scipy.sparse as sp
from sklearn.utils.fixes import loguniform
from sklearn.utils.validation import check_random_state

from cca_zoo.data.simulated import LinearSimulatedData
from cca_zoo.model_selection import GridSearchCV, RandomizedSearchCV
from cca_zoo.models import (
    CCA,
    GCCA,
    GRCCA,
    KCCA,
    MCCA,
    NCCA,  # SCCA_ADMM,
    PLS,
    PRCCA,
    SCCA_IPLS,
    SCCA_PMD,
    AltMaxVar,
    ElasticCCA,
    PartialCCA,
    SCCA_Parkhomenko,
    SCCA_Span,
    rCCA,
)

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


def test_linear_simulated_data():
    sim_data = LinearSimulatedData([10, 10]).sample(100)
    assert CCA().fit(sim_data).score(sim_data) > 0.9


def test_regularized_methods():
    # Test that linear regularized methods match PLS solution when using maximum regularisation.
    latent_dims = 2
    c = 1
    rcca = rCCA(latent_dims=latent_dims, c=[c, c]).fit([X, Y])
    mcca = MCCA(latent_dims=latent_dims, c=[c, c]).fit([X, Y])
    pls = PLS(latent_dims=latent_dims).fit([X, Y])
    gcca = GCCA(latent_dims=latent_dims, c=[c, c]).fit([X, Y])
    kernel = KCCA(latent_dims=latent_dims, c=[c, c], kernel=["linear", "linear"]).fit(
        (X, Y)
    )
    corr_gcca = gcca.score((X, Y))
    corr_mcca = mcca.score((X, Y))
    corr_kernel = kernel.score((X, Y))
    corr_pls = pls.score((X, Y))
    corr_rcca = rcca.score((X, Y))
    # Check the correlations from each unregularized method are the same
    assert np.testing.assert_array_almost_equal(corr_pls, corr_mcca, decimal=1) is None
    assert (
        np.testing.assert_array_almost_equal(corr_pls, corr_kernel, decimal=1) is None
    )
    assert np.testing.assert_array_almost_equal(corr_pls, corr_rcca, decimal=1) is None


def test_sparse_methods():
    tau1 = [5e-1]
    tau2 = [5e-1]
    param_grid = {"tau": [tau1, tau2]}
    pdd_cv = GridSearchCV(
        AltMaxVar(proximal="L0", random_state=rng), param_grid=param_grid
    ).fit([X, Y])
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
    tau1 = [0.1]
    tau2 = [0.1]
    param_grid = {"tau": [tau1, tau2]}
    pmd_cv = GridSearchCV(SCCA_PMD(random_state=rng), param_grid=param_grid).fit([X, Y])
    tau1 = [2e-1]
    tau2 = [2e-1]
    param_grid = {"tau": [tau1, tau2]}
    parkhomenko_cv = GridSearchCV(
        SCCA_Parkhomenko(random_state=rng), param_grid=param_grid
    ).fit([X, Y])
    tau1 = [2e-2]
    tau2 = [1e-2]
    param_grid = {"tau": [tau1, tau2]}
    # admm_cv = GridSearchCV(SCCA_ADMM(random_state=rng), param_grid=param_grid).fit(
    #     [X, Y]
    # )
    assert (pdd_cv.best_estimator_.weights[0] == 0).sum() > 0
    assert (pdd_cv.best_estimator_.weights[1] == 0).sum() > 0
    assert (pmd_cv.best_estimator_.weights[0] == 0).sum() > 0
    assert (pmd_cv.best_estimator_.weights[1] == 0).sum() > 0
    assert (scca_cv.best_estimator_.weights[0] == 0).sum() > 0
    assert (scca_cv.best_estimator_.weights[1] == 0).sum() > 0
    # assert (admm_cv.best_estimator_.weights[0] == 0).sum() > 0
    # assert (admm_cv.best_estimator_.weights[1] == 0).sum() > 0
    assert (parkhomenko_cv.best_estimator_.weights[0] == 0).sum() > 0
    assert (parkhomenko_cv.best_estimator_.weights[1] == 0).sum() > 0
    assert (elastic_cv.best_estimator_.weights[0] == 0).sum() > 0
    assert (elastic_cv.best_estimator_.weights[1] == 0).sum() > 0


def test_weighted_GCCA_methods():
    # TODO we have view weighted GCCA and missing observation GCCA
    latent_dims = 2
    c = 0
    unweighted_gcca = GCCA(latent_dims=latent_dims, c=[c, c]).fit([X, Y])
    deweighted_gcca = GCCA(
        latent_dims=latent_dims, c=[c, c], view_weights=[0.5, 0.5]
    ).fit([X, Y])
    corr_unweighted_gcca = unweighted_gcca.score((X, Y))
    corr_deweighted_gcca = deweighted_gcca.score((X, Y))
    # Check the correlations from each unregularized method are the same
    K = np.ones((2, X.shape[0]))
    K[0, 200:] = 0
    unobserved_gcca = GCCA(latent_dims=latent_dims, c=[c, c]).fit((X, Y), K=K)
    assert (
        np.testing.assert_array_almost_equal(
            corr_unweighted_gcca, corr_deweighted_gcca, decimal=1
        )
        is None
    )


def test_NCCA():
    latent_dims = 1
    ncca = NCCA(latent_dims=latent_dims).fit((X, Y))
    corr_ncca = ncca.score((X, Y))
    assert corr_ncca > 0.9


def test_l0():
    span_cca = SCCA_Span(
        latent_dims=1, regularisation="l0", tau=[2, 2], random_state=rng
    ).fit([X, Y])
    # swcca = SWCCA(tau=[5, 5], sample_support=5, random_state=rng).fit([X, Y])
    assert (np.abs(span_cca.weights[0]) > 1e-5).sum() == 2
    assert (np.abs(span_cca.weights[1]) > 1e-5).sum() == 2
    # assert (np.abs(swcca.weights[0]) > 1e-5).sum() == 5
    # assert (np.abs(swcca.weights[1]) > 1e-5).sum() == 5
    # assert (np.abs(swcca.sample_weights) > 1e-5).sum() == 5


def test_partialcca():
    # Tests that partial CCA scores are not correlated with partials
    pcca = PartialCCA(latent_dims=3)
    pcca.fit((X, Y), partials=Z)
    assert np.allclose(
        np.corrcoef(pcca.transform((X, Y), partials=Z)[0], Z, rowvar=False)[:3, :3]
        - np.eye(3),
        0,
        atol=0.001,
    )


def test_PRCCA():
    # Test that PRCCA works
    prcca = PRCCA(latent_dims=2, c=[0, 0]).fit(
        (X, Y), idxs=(np.arange(10), np.arange(11))
    )
    cca = CCA(latent_dims=2).fit([X, Y])
    assert (
        np.testing.assert_array_almost_equal(
            cca.score((X, Y)), prcca.score((X, Y)), decimal=1
        )
        is None
    )
    prcca = PRCCA(latent_dims=2, c=[1, 1]).fit(
        (X, Y), idxs=(np.arange(10), np.arange(11))
    )
    pls = PLS(latent_dims=2).fit([X, Y])
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
    grcca = GRCCA(latent_dims=1, c=[100, 0], mu=0).fit(
        (X, Y), feature_groups=[feature_group_1, feature_group_2]
    )
    grcca.score((X, Y))
    grcca.transform((X, Y))
    grcca = GRCCA(c=[100, 0, 50]).fit(
        (X, Y, Z), feature_groups=[feature_group_1, feature_group_2, feature_group_3]
    )
