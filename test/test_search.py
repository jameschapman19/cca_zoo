
import pytest
import numpy as np
from cca_zoo.model_selection import GridSearchCV
from cca_zoo.model_selection._search import param2grid
from cca_zoo.nonparametric import KCCA

# Tests for param2grid function
@pytest.mark.parametrize(
    "params, expected",
    [
        ({"c": [3, 4]}, [{"c": 3}, {"c": 4}]),
        (
            {"regs": [[1, 2], [3, 4]]},
            [{"regs": [1, 3]}, {"regs": [1, 4]}, {"regs": [2, 3]}, {"regs": [2, 4]}],
        ),
    ],
)
def test_param2grid(params, expected):
    grid = list(param2grid(params))
    assert grid == expected


def test_param2grid_with_iterable_types():
    # Given a dictionary of parameters with different iterable types
    params = {
        "alpha": [np.array([0.1, 0.01]), np.array([0.1, 0.01])],
        "regs": [(1, 2), (3, 4)],
    }
    # Expected result
    expected = [
    {"alpha": [0.1, 0.1], "regs": [1, 3]},
    {"alpha": [0.1, 0.1], "regs": [1, 4]},
    {"alpha": [0.1, 0.1], "regs": [2, 3]},
    {"alpha": [0.1, 0.1], "regs": [2, 4]},
    {"alpha": [0.1, 0.01], "regs": [1, 3]},
    {"alpha": [0.1, 0.01], "regs": [1, 4]},
    {"alpha": [0.1, 0.01], "regs": [2, 3]},
    {"alpha": [0.1, 0.01], "regs": [2, 4]},
    {"alpha": [0.01, 0.1], "regs": [1, 3]},
    {"alpha": [0.01, 0.1], "regs": [1, 4]},
    {"alpha": [0.01, 0.1], "regs": [2, 3]},
    {"alpha": [0.01, 0.1], "regs": [2, 4]},
    {"alpha": [0.01, 0.01], "regs": [1, 3]},
    {"alpha": [0.01, 0.01], "regs": [1, 4]},
    {"alpha": [0.01, 0.01], "regs": [2, 3]},
    {"alpha": [0.01, 0.01], "regs": [2, 4]},
]
    grid = list(param2grid(params))
    assert grid == expected


@pytest.fixture
def random_data():
    X = np.random.normal(size=(100, 10))
    Y = np.random.normal(size=(100, 10))
    return X, Y


def test_kernel_tuning(random_data):
    X, Y = random_data
    # Define a parameter grid
    param_grid = {
        "kernel": ["poly"],
        "c": [[1e-1], [1e-1, 2e-1]],
        "degree": [[2], [2, 3]],
    }

    # Test the grid search for KCCA
    kernel_reg = GridSearchCV(
        KCCA(latent_dimensions=1), param_grid=param_grid, cv=2, verbose=True
    ).fit([X, Y])

    # Additional assertions can be added to test properties of kernel_reg
    # For example:
    assert hasattr(kernel_reg, "best_estimator_")
    # assert kernel_reg.best_estimator_.kernel == "poly"