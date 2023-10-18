import pytest
from cca_zoo.visualisation import (
    CovarianceHeatmapDisplay,
    CorrelationHeatmapDisplay,
    ScoreScatterDisplay,
    WeightHeatmapDisplay,
    ExplainedVarianceDisplay,
    ExplainedCovarianceDisplay,
)
import numpy as np
from cca_zoo.linear import MCCA
import matplotlib.pyplot as plt

from cca_zoo.visualisation.scores import (
    JointScoreScatterDisplay,
    SeparateScoreScatterDisplay,
    SeparateJointScoreDisplay,
    PairScoreScatterDisplay,
)
from cca_zoo.visualisation.tsne_scores import TSNEScoreDisplay
from cca_zoo.visualisation.umap_scores import UMAPScoreDisplay


@pytest.fixture(scope="module")
def setup_data():
    X = np.random.rand(100, 10)
    Y = np.random.rand(100, 10)
    X -= X.mean(axis=0)
    Y -= Y.mean(axis=0)

    X_train, X_test = X[:50], X[50:]
    Y_train, Y_test = Y[:50], Y[50:]

    views = [X_train, Y_train]
    test_views = [X_test, Y_test]

    mcca = MCCA(latent_dimensions=2)
    mcca.fit(views)
    mcca.score(views)

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


def test_score_plot(setup_data):
    mcca, views, test_views = setup_data
    ScoreScatterDisplay.from_estimator(
        mcca, views, test_views=test_views, ax_labels=["Brain", "Behaviour"]
    ).plot()
    plt.close()


def test_score_plot_separate(setup_data):
    mcca, views, test_views = setup_data
    SeparateScoreScatterDisplay.from_estimator(
        mcca, views, test_views=test_views, ax_labels=["Brain", "Behaviour"]
    ).plot()
    plt.close()


def test_joint_score_plot(setup_data):
    mcca, views, test_views = setup_data
    JointScoreScatterDisplay.from_estimator(
        mcca, views, test_views=test_views, ax_labels=["Brain", "Behaviour"]
    ).plot()
    plt.close()


def test_joint_score_plot_separate(setup_data):
    mcca, views, test_views = setup_data
    SeparateJointScoreDisplay.from_estimator(
        mcca, views, test_views=test_views, ax_labels=["Brain", "Behaviour"]
    ).plot()
    plt.close()


def test_pairplot(setup_data):
    mcca, views, test_views = setup_data
    PairScoreScatterDisplay.from_estimator(mcca, views, test_views=test_views).plot()
    plt.close()


def test_covariance_heatmap_plot(setup_data):
    mcca, views, test_views = setup_data
    CovarianceHeatmapDisplay.from_estimator(mcca, views, test_views=test_views).plot()
    plt.close()


def test_correlation_heatmap_plot(setup_data):
    mcca, views, test_views = setup_data
    CorrelationHeatmapDisplay.from_estimator(mcca, views, test_views=test_views).plot()
    plt.close()


def test_biplots(setup_data):
    from cca_zoo.visualisation.biplot import WeightsBiPlotDisplay, LoadingsBiPlotDisplay

    mcca, views, test_views = setup_data
    WeightsBiPlotDisplay.from_estimator(mcca).plot()
    plt.close()
    LoadingsBiPlotDisplay.from_estimator(mcca, views, test_views=test_views).plot()
    plt.close()


def test_tsne_plot(setup_data):
    mcca, views, test_views = setup_data
    TSNEScoreDisplay.from_estimator(mcca, views, test_views=test_views).plot()
    plt.close()


def test_umap_plot(setup_data):
    mcca, views, test_views = setup_data
    UMAPScoreDisplay.from_estimator(mcca, views, test_views=test_views).plot()
    plt.close()
