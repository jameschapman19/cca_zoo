import numpy as np
import pytest
from matplotlib import pyplot as plt

from cca_zoo.linear import PLS
from cca_zoo.visualisation import Plotter


# Define a fixture for the sample data
@pytest.fixture
def sample_data():
    train_views = [np.random.randn(100, 10), np.random.randn(100, 10)]
    test_views = [np.random.randn(50, 10), np.random.randn(50, 10)]
    train_labels = np.random.choice([0, 1], 100)
    test_labels = np.random.choice([0, 1], 50)
    view_names = ["Brain", "Behavior"]
    title = "Sample Plot"
    model = PLS(latent_dimensions=5).fit(train_views)
    brain_weights = model.weights[0]
    behaviour_weights = model.weights[1]
    train_scores = model.transform(train_views)
    test_scores = model.transform(test_views)
    return {
        "train_views": train_views,
        "test_views": test_views,
        "train_scores": train_scores,
        "test_scores": test_scores,
        "train_labels": train_labels,
        "test_labels": test_labels,
        "view_names": view_names,
        "title": title,
        "brain_weights": brain_weights,
        "behaviour_weights": behaviour_weights,
        "model": model,
    }


# Test the plot_covariance_heatmap method with default parameters
def test_plot_covariance_heatmap_default(sample_data):
    plotter = Plotter()
    axs = plotter.plot_covariance_heatmap(sample_data["train_scores"], sample_data["test_scores"])

    # Check that the plot has the correct labels and title
    assert axs[0].get_title() == "Train Covariances"
    assert axs[1].get_title() == "Test Covariances"


# Test the plot_weights_heatmap method with default parameters
def test_plot_weights_heatmap_default(sample_data):
    plotter = Plotter()
    axs = plotter.plot_weights_heatmap(sample_data["brain_weights"], sample_data["behaviour_weights"])

    # Check that the plot has the correct labels and title
    assert axs[0].get_title() == "View 1 weights"
    assert axs[1].get_title() == "View 2 weights"


# Test the plot_explained_covariance method with default parameters
def test_plot_explained_covariance_default(sample_data):
    plotter = Plotter()
    ax = plotter.plot_explained_covariance(sample_data["model"], train_views=sample_data["train_views"], test_views=sample_data["test_views"])
