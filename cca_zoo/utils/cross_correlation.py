import numpy as np


def cross_corrcoef(A, B, rowvar=False):
    """Cross correlation of two matrices.

    Args:
        A (np.ndarray): Matrix of size (n,p).
        B (np.ndarray): Matrix of size (n,q).
        rowvar (bool, optional): Whether to calculate the correlation along the rows. Defaults to False.

    Returns:
        np.ndarray: Matrix of size (p,q) containing the cross correlation of A and B.
    """
    if rowvar:
        A = A.T
        B = B.T
    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)
    C = A.T @ B
    A = np.sqrt(np.sum(A**2, axis=0))
    B = np.sqrt(np.sum(B**2, axis=0))
    return C / (A * B)


def cross_cov(A, B, rowvar=False):
    """Cross covariance of two matrices.

    Args:
        A (np.ndarray): Matrix of size (n,p).
        B (np.ndarray): Matrix of size (n,q).
        rowvar (bool, optional): Whether to calculate the covariance along the rows. Defaults to False.

    Returns:
        np.ndarray: Matrix of size (p,q) containing the cross covariance of A and B.
    """
    if rowvar:
        A = A.T
        B = B.T
    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)
    C = A.T @ B
    return C / A.shape[0]
