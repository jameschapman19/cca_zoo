import numpy as np

from cca_zoo.model_selection import GridSearchCV
from cca_zoo.model_selection._search import param2grid


def test_param2grid_simple():
    # given a dictionary of parameters with lists for each view
    params = {"c": [3, 4]}
    # when we convert it to a parameter grid
    grid = list(param2grid(params))
    # then we expect a list of dictionaries with one value per view
    expected = [{"c": 3}, {"c": 4}]
    assert grid == expected


def test_param2grid():
    # given a dictionary of parameters with lists for each view
    params = {"regs": [[1, 2], [3, 4]]}
    # when we convert it to a parameter grid
    grid = list(param2grid(params))
    # then we expect a list of dictionaries with one value per view
    expected = [{"regs": [1, 3]}, {"regs": [1, 4]}, {"regs": [2, 3]}, {"regs": [2, 4]}]
    assert grid == expected


def test_param2grid_with_iterable_types():
    # given a dictionary of parameters with different iterable types
    params = {
        "alpha": [np.array([0.1, 0.01]), np.array([0.1, 0.01])],
        "regs": [(1, 2), (3, 4)],
    }
    # when we convert it to a parameter grid
    grid = list(param2grid(params))
    # then we expect a list of dictionaries with one value per view

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

    assert grid == expected


def test_kernel_tuning():
    X = np.random.normal(size=(100, 10))
    Y = np.random.normal(size=(100, 10))
    from cca_zoo.nonparametric import KCCA

    # We define a parameter grid with the polynomial kernel and different values for the regularization parameter (c) and the degree of the polynomial
    param_grid = {
        "kernel": ["poly"],
        "c": [[1e-1], [1e-1, 2e-1]],
        "degree": [[2], [2, 3]],
    }

    # We use GridSearchCV to find the best KCCA model with the polynomial kernel
    kernel_reg = GridSearchCV(
        KCCA(latent_dimensions=1), param_grid=param_grid, cv=2, verbose=True
    ).fit([X, Y])
