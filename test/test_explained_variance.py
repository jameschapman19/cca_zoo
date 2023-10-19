import numpy as np
import pytest

from cca_zoo._base import BaseModel
from cca_zoo.linear import MPLS

N = 50
features = [4, 6, 8]


@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)


@pytest.fixture
def toy_model(rng):
    model = BaseModel()
    model.weights = [rng.random((10, 3)), rng.random((8, 3)), rng.random((5, 3))]
    return model


@pytest.fixture
def synthetic_views(rng):
    # Generating three synthetic representations with N samples each
    view1 = rng.random((N, features[0]))
    view2 = rng.random((N, features[1]))
    view3 = rng.random((N, features[2]))
    # demean
    view1 -= view1.mean(axis=0)
    view2 -= view2.mean(axis=0)
    view3 -= view3.mean(axis=0)
    return [view1, view2, view3]


def test_explained_variance_ratio(toy_model, synthetic_views):
    explained_variance_ratios = toy_model.explained_variance_ratio(synthetic_views)

    # Verify if the ratios are between 0 and 1 for each latent dimension in each view
    for ratios in explained_variance_ratios:
        for ratio in ratios:
            assert (
                0 <= ratio <= 1
            ), f"Explained variance ratio should be between 0 and 1, but got {ratio}"


def test_transformed_covariance_ratio(toy_model, synthetic_views):
    maximum_dimension = min([view.shape[1] for view in synthetic_views])
    pls = MPLS(latent_dimensions=maximum_dimension).fit(synthetic_views)
    pls_cov_ratios = pls.explained_covariance_ratio(synthetic_views)
    # sum of these should be 1 within a small tolerance
    assert np.isclose(
        np.sum(pls_cov_ratios), 1, atol=1e-2
    ), "Expected sum of ratios to be 1"

    cov_ratios = toy_model.explained_covariance_ratio(synthetic_views)

    # Verify if the ratios are between 0 and 1 for each latent dimension in each view
    for ratio in cov_ratios:
        assert (
            0 <= ratio <= 1
        ), f"Explained covariance ratio should be between 0 and 1, but got {ratio}"


def test_explained_variance(toy_model, synthetic_views):
    explained_vars = toy_model.explained_variance(synthetic_views)
    assert all(
        isinstance(var, np.ndarray) for var in explained_vars
    ), "Expected numpy arrays"
    assert all(var.ndim == 1 for var in explained_vars), "Expected 1-dimensional arrays"


def test_explained_variance_cumulative(toy_model, synthetic_views):
    cumulative_ratios = toy_model.explained_variance_cumulative(synthetic_views)
    # Verifying if the ratios are increasing for each latent dimension in each view
    for ratios in cumulative_ratios:
        assert np.all(
            np.diff(ratios) >= 0
        ), "Expected cumulative ratios to be non-decreasing"


def test_explained_covariance(toy_model, synthetic_views):
    explained_covariances = toy_model.explained_covariance(synthetic_views)
    assert isinstance(explained_covariances, np.ndarray), "Expected a numpy array"
    assert explained_covariances.ndim == 1, "Expected 1-dimensional array"


def test_explained_covariance_ratio(toy_model, synthetic_views):
    explained_covariance_ratios = toy_model.explained_covariance_ratio(synthetic_views)
    # Verifying if the ratios are between 0 and 1 for each latent dimension in each view
    for ratio in explained_covariance_ratios:
        assert (
            0 <= ratio <= 1
        ), f"Explained covariance ratio should be between 0 and 1, but got {ratio}"


def test_explained_covariance_cumulative(toy_model, synthetic_views):
    cumulative_ratios = toy_model.explained_covariance_cumulative(synthetic_views)
    # Verifying if the ratios are increasing for each latent dimension in each view
    for ratios in cumulative_ratios:
        assert np.all(
            np.diff(ratios) >= 0
        ), "Expected cumulative ratios to be non-decreasing"
