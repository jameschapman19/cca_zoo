"""Linear algebra utilities: whitening, eigendecomposition, deflation."""

from __future__ import annotations

import numpy as np
import scipy.linalg


def svd_whiten(
    X: np.ndarray,
    regularization: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Whiten X using a regularised SVD decomposition.

    Computes W such that X @ W has covariance approximately equal to the
    identity matrix (or a regularised version thereof).

    Args:
        X: Array of shape (n_samples, n_features), assumed mean-centred.
        regularization: Ridge parameter in [0, 1].  0 gives full PCA whitening;
            1 gives identity (no whitening).

    Returns:
        Tuple ``(X_white, W)`` where ``X_white = X @ W`` and ``W`` is the
        (n_features, rank) whitening matrix.
    """
    n = X.shape[0]
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


def gevp(
    A: np.ndarray,
    B: np.ndarray | None,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve a symmetric (generalised) eigenvalue problem and return the top-k pairs.

    Solves ``A v = lambda B v`` (or the standard problem when B is None) and
    returns the k eigenpairs with the largest eigenvalues.

    Args:
        A: Symmetric matrix of shape (p, p).
        B: Symmetric positive-definite matrix of shape (p, p), or None for the
            standard eigenvalue problem.
        k: Number of eigenpairs to return.

    Returns:
        Tuple ``(eigvals, eigvecs)`` where ``eigvals`` has shape ``(k,)`` and
        ``eigvecs`` has shape ``(p, k)``, sorted in descending order.
    """
    p = A.shape[0]
    k_clamped = min(k, p)
    if B is None:
        eigvals, eigvecs = scipy.linalg.eigh(A, subset_by_index=[p - k_clamped, p - 1])
    else:
        eigvals, eigvecs = scipy.linalg.eigh(
            A, B, subset_by_index=[p - k_clamped, p - 1]
        )
    idx = np.argsort(eigvals)[::-1]
    return eigvals[idx].real, eigvecs[:, idx].real


def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """Apply element-wise soft (shrinkage) thresholding.

    Computes ``sign(x) * max(|x| - threshold, 0)``.

    Args:
        x: Input array.
        threshold: Non-negative threshold value.

    Returns:
        Thresholded array of the same shape as ``x``.
    """
    return np.asarray(np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0))


def deflate(
    views: list[np.ndarray],
    weights: list[np.ndarray],
) -> list[np.ndarray]:
    """Deflate views by removing the variance explained by current weights.

    Uses the Gram-Schmidt / projection deflation approach:
    ``X_deflated = X - (X @ w) (X @ w)^T X / ||(X @ w)||^2``

    Args:
        views: List of arrays each of shape (n_samples, n_features_i).
        weights: List of weight vectors each of shape (n_features_i, 1) or
            (n_features_i,).

    Returns:
        List of deflated arrays with the same shapes as ``views``.
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
