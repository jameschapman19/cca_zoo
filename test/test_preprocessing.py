import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from cca_zoo.preprocessing import MultiViewPreprocessing


@pytest.fixture
def mock_data():
    brain_data = np.random.rand(100, 10)
    behavior_data = np.random.rand(100, 10)
    groups = np.random.randint(0, 2, 100)
    return brain_data, behavior_data, groups


def test_preprocessing_steps_are_fitted_after_fit(mock_data):
    brain_data, behavior_data, _ = mock_data
    views = [brain_data, behavior_data]
    preprocessing_steps = [StandardScaler(), StandardScaler()]

    mvp = MultiViewPreprocessing(preprocessing_steps)
    mvp.fit(views)

    assert all(
        [
            hasattr(preprocessor, "mean_") and hasattr(preprocessor, "var_")
            for preprocessor in mvp.preprocessing_list
        ]
    )


def test_transformed_views_have_same_shape_as_original(mock_data):
    brain_data, behavior_data, _ = mock_data
    views = [brain_data, behavior_data]
    preprocessing_steps = [StandardScaler(), StandardScaler()]

    mvp = MultiViewPreprocessing(preprocessing_steps)
    mvp.fit(views)
    transformed_views = mvp.transform(views)

    assert all(
        [
            original.shape == transformed.shape
            for original, transformed in zip(views, transformed_views)
        ]
    )


def test_transformed_views_have_zero_mean_and_unit_variance(mock_data):
    brain_data, behavior_data, _ = mock_data
    views = [brain_data, behavior_data]
    preprocessing_steps = [StandardScaler(), StandardScaler()]

    mvp = MultiViewPreprocessing(preprocessing_steps)
    mvp.fit(views)
    transformed_views = mvp.transform(views)

    assert all(
        [
            np.allclose(transformed.mean(axis=0), 0)
            and np.allclose(transformed.std(axis=0), 1)
            for transformed in transformed_views
        ]
    )


def test_no_preprocessing_steps_does_not_change_data(mock_data):
    brain_data, behavior_data, _ = mock_data
    views = [brain_data, behavior_data]
    preprocessing_steps = [None, None]

    mvp = MultiViewPreprocessing(preprocessing_steps)
    mvp.fit(views)
    transformed_views = mvp.transform(views)

    assert all(
        [
            np.array_equal(original, transformed)
            for original, transformed in zip(views, transformed_views)
        ]
    )
