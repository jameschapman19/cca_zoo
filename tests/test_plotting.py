import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from cca_zoo.linear import MCCA
from cca_zoo.visualisation import (
    CovarianceHeatmapDisplay,
    CorrelationHeatmapDisplay,
    RepresentationScatterDisplay,
    WeightHeatmapDisplay,
    ExplainedVarianceDisplay,
    ExplainedCovarianceDisplay,
    TSNERepresentationDisplay,
    UMAPRepresentationDisplay,
)
from cca_zoo.visualisation.representations import (
    JointRepresentationScatterDisplay,
    SeparateRepresentationScatterDisplay,
    SeparateJointRepresentationDisplay,
    PairRepresentationScatterDisplay,
)


@pytest.fixture(scope="module")
def setup_data():
    X = np.random.rand(10, 3)
    Y = np.random.rand(10, 3)
    X -= X.mean(axis=0)
    Y -= Y.mean(axis=0)

    # train test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

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
    RepresentationScatterDisplay.from_estimator(
        mcca, views, test_views=test_views, ax_labels=["Brain", "Behaviour"]
    ).plot()
    plt.close()


def test_score_plot_separate(setup_data):
    mcca, views, test_views = setup_data
    SeparateRepresentationScatterDisplay.from_estimator(
        mcca, views, test_views=test_views, ax_labels=["Brain", "Behaviour"]
    ).plot()
    plt.close()


def test_joint_score_plot(setup_data):
    mcca, views, test_views = setup_data
    JointRepresentationScatterDisplay.from_estimator(
        mcca, views, test_views=test_views, ax_labels=["Brain", "Behaviour"]
    ).plot()
    plt.close()


def test_joint_score_plot_separate(setup_data):
    mcca, views, test_views = setup_data
    SeparateJointRepresentationDisplay.from_estimator(
        mcca, views, test_views=test_views, ax_labels=["Brain", "Behaviour"]
    ).plot()
    plt.close()


def test_pairplot(setup_data):
    mcca, views, test_views = setup_data
    PairRepresentationScatterDisplay.from_estimator(
        mcca, views, test_views=test_views
    ).plot()
    plt.close()


def test_covariance_heatmap_plot(setup_data):
    mcca, views, test_views = setup_data
    CovarianceHeatmapDisplay.from_estimator(mcca, views, test_views=test_views).plot()
    plt.close()


def test_correlation_heatmap_plot(setup_data):
    mcca, views, test_views = setup_data
    CorrelationHeatmapDisplay.from_estimator(mcca, views, test_views=test_views).plot()
    plt.close()


# def test_biplots(setup_data):
#     from cca_zoo.visualisation.biplot import WeightsBiPlotDisplay, LoadingsBiPlotDisplay
#
#     mcca, views, test_views = setup_data
#     WeightsBiPlotDisplay.from_estimator(mcca).plot()
#     plt.close()
#     LoadingsBiPlotDisplay.from_estimator(mcca, views, test_views=test_views).plot()
#     plt.close()


def test_tsne_plot(setup_data):
    mcca, views, test_views = setup_data
    TSNERepresentationDisplay.from_estimator(mcca, views, test_views=test_views).plot()
    plt.close()


def test_umap_plot(setup_data):
    mcca, views, test_views = setup_data
    UMAPRepresentationDisplay.from_estimator(mcca, views, test_views=test_views).plot()
    plt.close()
