"""Linear algebra utilities: whitening, eigendecomposition, deflation."""

from __future__ import annotations

import string

import numpy as np
import scipy.linalg


def svd_whiten(
    X: np.ndarray,
    regularization: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Whiten ``X`` with a ridge-regularised covariance.

    Uses an eigendecomposition of the covariance when ``n >= p`` and an SVD
    of ``X`` otherwise.

    Args:
        X: Centred array of shape (n_samples, n_features).
        regularization: Ridge blend in ``[0, 1]``; 0 is PCA whitening and 1
            no whitening.

    Returns:
        ``(X @ W, W)``, with ``W`` of shape (n_features, rank).
    """
    n, p = X.shape
    if n >= p:
        # Covariance path -- avoids the large n x p matrix U from thin SVD.
        C = X.T @ X / (n - 1)
        lam, V = np.linalg.eigh(C)
        # A strict ``lam > 0`` is not numerically safe here: forming X.T @ X
        # squares the condition number, so near-zero eigenvalues of
        # rank-deficient data can carry noise of either sign and slip
        # through a zero threshold. Use a relative tolerance instead,
        # matching numpy.linalg.matrix_rank's convention.
        tol = lam.max() * p * np.finfo(lam.dtype).eps
        pos = lam > tol
        lam, V = lam[pos], V[:, pos]
        inv_sqrt = 1.0 / np.sqrt((1.0 - regularization) * lam + regularization)
        W = V * inv_sqrt
        X_white = X @ W
    else:
        # SVD path -- avoids forming the p x p covariance matrix.
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        # Keep only dimensions with positive singular values
        pos = s > 0
        s = s[pos]
        U = U[:, pos]
        Vt = Vt[pos, :]
        # Eigenvalues of the sample covariance
        lam = s**2 / (n - 1)
        # Regularised inverse square root: ((1 - c) * lam + c)^{-1/2}
        inv_sqrt = 1.0 / np.sqrt((1.0 - regularization) * lam + regularization)
        # Whitening matrix: shape (n_features, rank)
        W = Vt.T * inv_sqrt
        X_white = U * (s * inv_sqrt)
    return X_white, W


def psd_inverse_sqrt(matrix: np.ndarray, floor: float) -> np.ndarray:
    """Inverse square root of a symmetric matrix, shifted to be positive definite.

    The spectrum is raised so its smallest eigenvalue is at least ``floor``.

    Args:
        matrix: Symmetric matrix of shape (p, p).
        floor: Smallest eigenvalue allowed.

    Returns:
        Symmetric matrix of shape (p, p).
    """
    eigenvalues, vectors = np.linalg.eigh(matrix)
    eigenvalues = eigenvalues + max(0.0, floor - eigenvalues[0])
    result: np.ndarray = (vectors / np.sqrt(eigenvalues)) @ vectors.T
    return result


def cross_moment_tensor(views: list[np.ndarray]) -> np.ndarray:
    """Mean over samples of the outer product of each view's row.

    For two views, ``views[0].T @ views[1] / n``.

    Args:
        views: Arrays of shape (n_samples, p_i).

    Returns:
        Array of shape (p_0, ..., p_{m-1}).
    """
    axes = string.ascii_letters[1 : len(views) + 1]
    subscripts = ",".join("a" + axis for axis in axes) + "->" + axes
    moment: np.ndarray = np.einsum(subscripts, *views, optimize=True) / len(views[0])
    return moment


def gevp(
    A: np.ndarray,
    B: np.ndarray | None,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Top ``k`` eigenpairs of ``A v = lambda B v``, or of ``A`` when ``B`` is None.

    Args:
        A: Symmetric matrix of shape (p, p).
        B: Symmetric positive-definite matrix of shape (p, p), or None.
        k: Number of eigenpairs.

    Returns:
        ``(eigvals, eigvecs)`` of shapes (k,) and (p, k), in descending order.
    """
    p = A.shape[0]
    k_clamped = min(k, p)
    eigvals, eigvecs = scipy.linalg.eigh(A, B, subset_by_index=[p - k_clamped, p - 1])
    idx = np.argsort(eigvals)[::-1]
    return eigvals[idx].real, eigvecs[:, idx].real


def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """Soft thresholding, ``sign(x) * max(|x| - threshold, 0)``."""
    return np.asarray(np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0))


def deflate(
    views: list[np.ndarray],
    weights: list[np.ndarray],
) -> list[np.ndarray]:
    """Remove from each view the variance along its current projection.

    ``X - (X w)(X w)' X / ||X w||^2``.

    Args:
        views: Arrays of shape (n_samples, n_features_i).
        weights: Weight vectors of shape (n_features_i,) or (n_features_i, 1).

    Returns:
        The deflated views.
    """
    deflated = []
    for view, w in zip(views, weights):
        w_col = w.reshape(-1, 1) if w.ndim == 1 else w[:, :1]
        score = view @ w_col  # (n, 1)
        norm_sq = float(np.squeeze(score.T @ score))
        if norm_sq > 1e-12:
            view = view - score @ (score.T @ view) / norm_sq
        deflated.append(view)
    return deflated
