import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from cca_zoo.preprocessing import MultiViewPreprocessing


@pytest.fixture
def data():
    # Load the data from the text files
    brain_df = np.random.rand(100, 10)
    behavior_df = np.random.rand(100, 10)
    groups = np.random.randint(0, 2, 100)
    return brain_df, behavior_df, groups


def test_multiview_preprocessing(data):
    # Test the MultiViewPreprocessing class
    brain_df, behavior_df, groups = data
    # Create a list of views with brain and behavior data
    views = [brain_df, behavior_df]
    # Create a list of preprocessing steps with standard scaling for each view
    preprocessing_list = [StandardScaler(), StandardScaler()]
    # Create a MultiViewPreprocessing instance with the preprocessing list
    mvp = MultiViewPreprocessing(preprocessing_list)
    # Fit the MultiViewPreprocessing to the views
    mvp.fit(views)
    # Check that the preprocessing steps are fitted for each view
    assert all(
        [
            hasattr(preprocessing, "mean_") and hasattr(preprocessing, "var_")
            for preprocessing in mvp.preprocessing_list
        ]
    )
    # Transform the views using the MultiViewPreprocessing
    transformed_views = mvp.transform(views)
    # Check that the transformed views have the same shape as the original views
    assert all(
        [
            view.shape == transformed_view.shape
            for view, transformed_view in zip(views, transformed_views)
        ]
    )
    # Check that the transformed views have zero mean and unit variance for each feature
    assert all(
        [
            np.allclose(transformed_view.mean(axis=0), 0)
            and np.allclose(transformed_view.std(axis=0), 1)
            for transformed_view in transformed_views
        ]
    )
