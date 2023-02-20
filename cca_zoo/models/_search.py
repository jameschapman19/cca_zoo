import numpy as np
from pyproximal.proximal.L1 import _softthreshold


def _bin_search(current, previous, current_val, previous_val, min_, max_):
    """
    Binary search helper function.

    Parameters
    ----------
    current : current parameter value
    previous :
    current_val :
    previous_val :
    min_ : minimum parameter value resulting in function value less than zero
    max_ : maximum parameter value resulting in function value greater than zero

    Returns
    -------
    new :
    current :
    min_ :
    max_ :

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


def _delta_search(w, c, init=0, tol=1e-9, max_iter=1000):
    """
    Searches for threshold delta such that the 1-norm of weights w is less than or equal to c
    Parameters
    ----------
    w : weights found by one power method iteration
    c : 1-norm threshold
    init : initial value for delta
    tol : tolerance for convergence

    Returns
    -------
    updated weights

    """
    # First normalise the weights unit length
    w = w / np.linalg.norm(w, 2)
    converged = False
    min_ = 0
    max_ = 10
    current = init
    previous = current
    previous_val = 0
    i = 0
    while not converged:
        i += 1
        coef = _softthreshold(w, current)
        if np.linalg.norm(coef) > 0:
            coef /= np.linalg.norm(coef)
        current_val = c - np.linalg.norm(coef, 1)
        current, previous, min_, max_ = _bin_search(
            current, previous, current_val, previous_val, min_, max_
        )
        if (np.abs(current_val) < tol) or i == max_iter:
            converged = True
        previous_val = current_val
    return coef


def support_threshold(data, support, **kwargs):
    idx = np.argpartition(data.ravel(), data.shape[0] - support)
    data[idx[:-support]] = 0
    return data
