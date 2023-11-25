import numpy as np
from scipy.optimize import minimize


def _delta_search(w, c, tol=1e-8):
    """
    Searches for threshold delta such that the 1-norm of weights_ w is less than or equal to c and the 2-norm is equal to 1.
    Parameters
    ----------
    w : numpy array
        weights_ found by one power method iteration
    c : float
        1-norm threshold
    init : float, optional
        initial value for delta (default is 0)

    Returns
    -------
    numpy array
        updated weights_

    """
    # First normalize the weights_ to unit length
    w = w / np.linalg.norm(w)

    # Define a scalar function that returns the difference between the 1-norm of coefficients and the threshold c
    def f(delta):
        # Apply soft thresholding to the weights_ with delta
        coef = np.clip(w - delta, 0, None) - np.clip(-w - delta, 0, None)

        if np.sum(coef**2) == 0:
            coef[:] = 1000
        else:
            # Normalize the coefficients to unit length if nonzero
            coef /= np.linalg.norm(coef)

        # Return the square of the difference between the 1-norm of coefficients and the threshold c
        return (np.sum(np.abs(coef)) - c) ** 2

    # Find the minimum of f using scipy minimization function
    # You can specify the method or let the function choose the best one for you
    # You can also pass other parameters like tol, maxiter, etc.
    # bound x to be between 0 and 1

    # try some different methods until one gets result.success == True
    result = minimize(f, x0=0, bounds=[(0, 1)], method="L-BFGS-B", tol=tol)
    if not result.success:
        result = minimize(f, x0=0, bounds=[(0, 1)], method="TNC", tol=tol)
    if not result.success:
        result = minimize(f, x0=0, bounds=[(0, 1)], method="SLSQP", tol=tol)
    if not result.success:
        result = minimize(f, x0=0, bounds=[(0, 1)], method="trust-constr", tol=tol)

    # Check if the solution is valid and converged
    if result.success:
        # Get the optimal delta from the result object
        delta = result.x

        # Apply soft thresholding to the weights_ with optimal delta
        coef = np.where(w - delta > 0, w - delta, 0) - np.where(
            -w - delta > 0, -w - delta, 0
        )

        # Normalize the coefficients to unit length if nonzero
        coef /= np.linalg.norm(coef)

        # Return updated weights_
        return coef

    else:
        # Raise an exception if no solution was found
        raise ValueError("No root was found for f")


def support_threshold(data, support, **kwargs):
    idx = np.argpartition(data.ravel(), data.shape[0] - support)
    data[idx[:-support]] = 0
    return data
