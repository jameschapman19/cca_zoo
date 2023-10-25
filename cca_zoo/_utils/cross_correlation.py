import numpy as np
import torch


def cross_corrcoef(A, B, rowvar=True):
    """Cross correlation of two matrices.

    Args:
        A (np.ndarray or torch.Tensor): Matrix of size (n, p).
        B (np.ndarray or torch.Tensor): Matrix of size (n, q).
        rowvar (bool, optional): Whether to calculate the correlation along the rows. Defaults to False.

    Returns:
        np.ndarray or torch.Tensor: Matrix of size (p, q) containing the cross correlation of A and B.
    """
    is_torch = torch.is_tensor(A) and torch.is_tensor(B)
    if rowvar == False:
        A = A.T
        B = B.T

    A = (
        A - A.mean(axis=1, keepdims=True)
        if not is_torch
        else A - A.mean(dim=1, keepdim=True)
    )
    B = (
        B - B.mean(axis=1, keepdims=True)
        if not is_torch
        else B - B.mean(dim=1, keepdim=True)
    )

    C = A @ B.T

    A = (
        np.sqrt(np.sum(A**2, axis=1))
        if not is_torch
        else torch.sqrt(torch.sum(A**2, dim=1))
    )
    B = (
        np.sqrt(np.sum(B**2, axis=1))
        if not is_torch
        else torch.sqrt(torch.sum(B**2, dim=1))
    )

    return C / np.outer(A, B) if not is_torch else C / torch.outer(A, B)


def cross_cov(A, B, rowvar=True, bias=False):
    """Cross covariance of two matrices.

    Args:
        A (np.ndarray or torch.Tensor): Matrix of size (n, p).
        B (np.ndarray or torch.Tensor): Matrix of size (n, q).
        rowvar (bool, optional): Whether to calculate the covariance along the rows. Defaults to False.

    Returns:
        np.ndarray or torch.Tensor: Matrix of size (p, q) containing the cross covariance of A and B.
    """
    is_torch = torch.is_tensor(A) and torch.is_tensor(B)
    if rowvar == False:
        A = A.T
        B = B.T

    A = (
        A - A.mean(axis=1, keepdims=True)
        if not is_torch
        else A - A.mean(dim=1, keepdim=True)
    )
    B = (
        B - B.mean(axis=1, keepdims=True)
        if not is_torch
        else B - B.mean(dim=1, keepdim=True)
    )

    C = A @ B.T

    if bias:
        return C / A.shape[1] if not is_torch else C / A.size(1)
    else:
        return C / (A.shape[1] - 1) if not is_torch else C / (A.size(1) - 1)
