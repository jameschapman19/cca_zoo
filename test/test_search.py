import numpy as np

from cca_zoo.model_selection._search import param2grid


def test_param2grid_simple():
    # given a dictionary of parameters with lists for each view
    params = {'c': [3, 4]}
    # when we convert it to a parameter grid
    grid = list(param2grid(params))
    # then we expect a list of dictionaries with one value per view
    expected = [{'c': 3}, {'c': 4}]
    assert grid == expected
def test_param2grid():
    # given a dictionary of parameters with lists for each view
    params = {'regs': [[1, 2], [3, 4]]}
    # when we convert it to a parameter grid
    grid = list(param2grid(params))
    # then we expect a list of dictionaries with one value per view
    expected = [{'regs': [1, 3]},
                {'regs': [1, 4]},
                {'regs': [2, 3]},
                {'regs': [2, 4]}]
    assert grid == expected

def test_param2grid_with_iterable_types():
    # given a dictionary of parameters with different iterable types
    params = {'alpha': [np.array([0.1, 0.01]), np.array([0.1, 0.01])],'regs': [(1, 2),(3,4) ]}
    # when we convert it to a parameter grid
    grid = list(param2grid(params))
    # then we expect a list of dictionaries with one value per view

    expected = [{'alpha': [0.1, 0.1], 'regs': [1, 3]},
                {'alpha': [0.1, 0.1], 'regs': [1, 4]},
                {'alpha': [0.1, 0.1], 'regs': [2, 3]},
                {'alpha': [0.1, 0.1], 'regs': [2, 4]},
                {'alpha': [0.1, 0.01], 'regs': [1, 3]},
                {'alpha': [0.1, 0.01], 'regs': [1, 4]},
                {'alpha': [0.1, 0.01], 'regs': [2, 3]},
                {'alpha': [0.1, 0.01], 'regs': [2, 4]},
                {'alpha': [0.01, 0.1], 'regs': [1, 3]},
                {'alpha': [0.01, 0.1], 'regs': [1, 4]},
                {'alpha': [0.01, 0.1], 'regs': [2, 3]},
                {'alpha': [0.01, 0.1], 'regs': [2, 4]},
                {'alpha': [0.01, 0.01], 'regs': [1, 3]},
                {'alpha': [0.01, 0.01], 'regs': [1, 4]},
                {'alpha': [0.01, 0.01], 'regs': [2, 3]},
                {'alpha': [0.01, 0.01], 'regs': [2, 4]}]



    assert grid == expected
