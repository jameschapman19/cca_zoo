import pytest
from cca_zoo.visualisation import (
    CovarianceHeatmapDisplay,
    CorrelationHeatmapDisplay,
    ScoreDisplay,
    WeightHeatmapDisplay,
    ExplainedVarianceDisplay,
    ExplainedCovarianceDisplay,
)
import numpy as np
from cca_zoo.linear import MCCA
import matplotlib.pyplot as plt


@pytest.fixture(scope="module")
def setup_data():
    X = np.random.rand(100, 10)
    Y = np.random.rand(100, 10)
    Z = np.random.rand(100, 10)

    X_train, X_test = X[:50], X[50:]
    Y_train, Y_test = Y[:50], Y[50:]
    Z_train, Z_test = Z[:50], Z[50:]

    views = [X_train, Y_train, Z_train]
    test_views = [X_test, Y_test, Z_test]

    mcca = MCCA(latent_dimensions=2)
    mcca.fit(views)

    return mcca, views, test_views


def test_explained_variance_plot(setup_data):
    mcca, views, test_views = setup_data
    ExplainedVarianceDisplay.from_estimator(mcca, views, test_views=test_views).plot()
    plt.close()


def test_explained_covariance_plot(setup_data):
    mcca, views, test_views = setup_data
    ExplainedCovarianceDisplay.from_estimator(mcca, views, test_views=test_views).plot()
    plt.close()


def test_weight_heatmap_plot(setup_data):
    mcca, _, _ = setup_data
    WeightHeatmapDisplay.from_estimator(mcca).plot()
    plt.close()


def test_score_heatmap_plot(setup_data):
    mcca, views, test_views = setup_data
    ScoreDisplay.from_estimator(mcca, views, test_views=test_views).plot()
    plt.close()


def test_covariance_heatmap_plot(setup_data):
    mcca, views, test_views = setup_data
    CovarianceHeatmapDisplay.from_estimator(mcca, views, test_views=test_views).plot()
    plt.close()


def test_correlation_heatmap_plot(setup_data):
    mcca, views, test_views = setup_data
    CorrelationHeatmapDisplay.from_estimator(mcca, views, test_views=test_views).plot()
    plt.close()
