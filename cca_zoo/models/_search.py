import numpy as np

from cca_zoo.models._proximal_operators import soft_threshold


def _bin_search(current, previous, current_val, previous_val, min_, max_):
    """Binary search helper function:
    current:current parameter value
    previous:previous parameter value
    current_val:current function value
    previous_val: previous function values
    min_:minimum parameter value resulting in function value less than zero
    max_:maximum parameter value resulting in function value greater than zero
    Problem needs to be set up so that greater parameter, greater target
    """
    if previous_val is None:
        previous_val = current_val
    if current_val <= 0:
        if previous_val <= 0:
            new = (current + max_) / 2
        if previous_val > 0:
            new = (current + previous) / 2
        if current > min_:
            min_ = current
    if current_val > 0:
        if previous_val > 0:
            new = (current + min_) / 2
        if previous_val <= 0:
            new = (current + previous) / 2
        if current < max_:
            max_ = current
    return new, current, min_, max_


def _delta_search(w, c, positive=False, init=0, tol=1e-9):
    """
    Searches for threshold delta such that the 1-norm of weights w is less than or equal to c
    :param w: weights found by one power method iteration
    :param c: 1-norm threshold
    :return: updated weights
    """
    # First normalise the weights unit length
    w = w / np.linalg.norm(w, 2)
    converged = False
    min_ = 0
    max_ = 10
    current = init
    previous = current
    previous_val = None
    i = 0
    while not converged:
        i += 1
        coef = soft_threshold(w, current)
        if np.linalg.norm(coef) > 0:
            coef /= np.linalg.norm(coef)
        current_val = c - np.linalg.norm(coef, 1)
        current, previous, min_, max_ = _bin_search(
            current, previous, current_val, previous_val, min_, max_
        )
        previous_val = current_val
        if np.abs(current_val) < tol or np.abs(max_ - min_) < tol or i == 150:
            converged = True
    return coef
