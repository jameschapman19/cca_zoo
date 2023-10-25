import pytest
from sklearn import clone
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.utils._testing import set_random_state
from sklearn.utils.estimator_checks import check_estimator

from cca_zoo.linear import (
    GCCA,
    CCA_EY,
    CCA_GHA,
    PLS_EY,
    PLSStochasticPower,
    GRCCA,
    PLS_ALS,
    SCCA_IPLS,
    SCCA_PMD,
    ElasticCCA,
    SCCA_Parkhomenko,
    SCCA_Span,
    CCA,
    MCCA,
    rCCA,
    PartialCCA,
    PCACCA,
    MPLS,
    PLS,
    PRCCA,
    TCCA,
)


class HackyDuplicator(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        X = self._validate_data(
            X,
            accept_sparse="csc",
        )

        return self

    def transform(self, X, y=None):
        return [X, X]


class HackyJoiner(TransformerMixin, BaseEstimator):
    def fit(self, views, y=None):
        X = self._validate_data(
            views[0],
            accept_sparse="csc",
        )
        return self

    def transform(self, views, y=None):
        return views[0]

random_state=0

@pytest.mark.parametrize(
    "estimator",
    [
        GCCA(random_state=random_state),
        CCA_EY(random_state=random_state),
        CCA_GHA(random_state=random_state),
        PLS_EY(random_state=random_state),
        PLSStochasticPower(random_state=random_state),
        GRCCA(random_state=random_state),
        PLS_ALS(random_state=random_state),
        SCCA_IPLS(random_state=random_state),
        SCCA_PMD(random_state=random_state),
        ElasticCCA(random_state=random_state),
        SCCA_Parkhomenko(random_state=random_state),
        SCCA_Span(random_state=random_state),
        CCA(random_state=random_state),
        MCCA(random_state=random_state),
        rCCA(random_state=random_state),
        PartialCCA(random_state=random_state),
        PCACCA(random_state=random_state),
        MPLS(random_state=random_state),
        PLS(random_state=random_state),
        PRCCA(random_state=random_state),
        TCCA(random_state=random_state),
    ],
)
def test_all_estimators(estimator):
    # hack to get around the two-view problem. We will make a pipeline with splitter
    estimator = Pipeline(
        [
            ("splitter", HackyDuplicator()),
            ("estimator", estimator),
            ("joiner", HackyJoiner()),
        ]
    )
    return check_estimator(estimator)
