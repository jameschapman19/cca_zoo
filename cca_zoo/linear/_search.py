import numpy as np

from scipy.optimize import root_scalar


def _delta_search(w, c, init=0):
    """
    Searches for threshold delta such that the 1-norm of weights w is less than or equal to c and the 2-norm is equal to 1.
    Parameters
    ----------
    w : numpy array
        weights found by one power method iteration
    c : float
        1-norm threshold
    init : float, optional
        initial value for delta (default is 0)

    Returns
    -------
    numpy array
        updated weights

    """
    # First normalize the weights to unit length
    w = w / np.sum(w ** 2) ** 0.5

    # Define a scalar function that returns the difference between the 1-norm of coefficients and the threshold c
    def f(delta):
        # Apply soft thresholding to the weights with delta
        coef = np.clip(w - delta, 0, None) - np.clip(-w - delta, 0, None)

        # Normalize the coefficients to unit length if nonzero
        if np.sum(coef ** 2) > 0:
            coef /= np.sum(coef ** 2) ** 0.5

        # Return the difference between the 1-norm of coefficients and the threshold c
        return c - np.sum(np.abs(coef))

    # Find the root of f using scipy root finding function
    # You can specify the method or let the function choose the best one for you
    # You can also pass other parameters like xtol, rtol, maxiter, etc.
    result = root_scalar(f, x0=init,x1=1, method="secant")

    # Check if the solution is valid and converged
    if result.converged:
        # Get the optimal delta from the result object
        delta = result.root

        # Apply soft thresholding to the weights with optimal delta
        coef = np.clip(w - delta, 0, None) - np.clip(-w - delta, 0, None)

        # Normalize the coefficients to unit length if nonzero
        if np.sum(coef ** 2) > 0:
            coef /= np.sum(coef ** 2) ** 0.5

        # Return updated weights
        return coef

    else:
        # Raise an exception if no solution was found
        raise ValueError("No root was found for f")


def support_threshold(data, support, **kwargs):
    idx = np.argpartition(data.ravel(), data.shape[0] - support)
    data[idx[:-support]] = 0
    return data
