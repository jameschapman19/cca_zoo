"""Linear algebra utilities: whitening, eigendecomposition, deflation.

The functions used by the Array API models (``covariance``, ``block_diag``,
``svd_whiten``, ``psd_inverse_sqrt`` and ``gevp``) work in the namespace of
their inputs, through scikit-learn's ``array_api_dispatch``.
"""

from __future__ import annotations

import string
from typing import Any

import numpy as np
from sklearn.utils._array_api import device, get_namespace


def covariance(X: Any) -> Any:
    """Sample covariance of the columns of ``X``, shape (p, p)."""
    xp, _ = get_namespace(X)
    centred = X - xp.mean(X, axis=0)
    return centred.T @ centred / (X.shape[0] - 1)


def block_diag(blocks: list[Any]) -> Any:
    """The square ``blocks`` along the diagonal of one matrix."""
    xp, _ = get_namespace(*blocks)
    total = sum(b.shape[0] for b in blocks)
    matrix = xp.zeros((total, total), dtype=blocks[0].dtype, device=device(blocks[0]))
    start = 0
    for b in blocks:
        end = start + b.shape[0]
        matrix[start:end, start:end] = b
        start = end
    return matrix


def svd_whiten(X: Any, regularization: float = 0.0) -> tuple[Any, Any]:
    """Whiten ``X`` with a ridge-regularised covariance.

    Uses an eigendecomposition of the covariance when ``n >= p`` and an SVD
    of ``X`` otherwise. Directions below the numerical rank are dropped, with
    ``numpy.linalg.matrix_rank``'s relative tolerance: centred data always has
    a null direction, and a strict zero threshold would keep it with an
    enormous weight.

    Args:
        X: Centred array of shape (n_samples, n_features).
        regularization: Ridge blend in ``[0, 1]``; 0 is PCA whitening and 1
            no whitening.

    Returns:
        ``(X @ W, W)``, with ``W`` of shape (n_features, rank).
    """
    xp, _ = get_namespace(X)
    n, p = X.shape
    if n >= p:
        lam, V = xp.linalg.eigh(X.T @ X / (n - 1))
        rank = int(xp.count_nonzero(lam > xp.max(lam) * p * xp.finfo(lam.dtype).eps))
        lam, V = lam[p - rank :], V[:, p - rank :]
    else:
        _, s, Vt = xp.linalg.svd(X, full_matrices=False)
        rank = int(xp.count_nonzero(s > xp.max(s) * max(n, p) * xp.finfo(s.dtype).eps))
        lam, V = s[:rank] ** 2 / (n - 1), Vt[:rank, :].T
    W = V / xp.sqrt((1.0 - regularization) * lam + regularization)
    return X @ W, W


def psd_inverse_sqrt(matrix: Any, floor: float) -> Any:
    """Inverse square root of a symmetric matrix, shifted to be positive definite.

    The spectrum is raised so its smallest eigenvalue is at least ``floor``.

    Args:
        matrix: Symmetric matrix of shape (p, p).
        floor: Smallest eigenvalue allowed.

    Returns:
        Symmetric matrix of shape (p, p).
    """
    xp, _ = get_namespace(matrix)
    eigenvalues, vectors = xp.linalg.eigh(matrix)
    eigenvalues = eigenvalues + max(0.0, floor - float(eigenvalues[0]))
    return (vectors / xp.sqrt(eigenvalues)) @ vectors.T


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


def gevp(A: Any, B: Any | None, k: int) -> tuple[Any, Any]:
    """Top ``k`` eigenpairs of ``A v = lambda B v``, or of ``A`` when ``B`` is None.

    The generalized problem is reduced by the Cholesky factor ``B = L L'`` to
    the symmetric ``L^{-1} A L^{-T} u = lambda u``, with ``v = L^{-T} u``.

    Args:
        A: Symmetric matrix of shape (p, p).
        B: Symmetric positive-definite matrix of shape (p, p), or None.
        k: Number of eigenpairs.

    Returns:
        ``(eigvals, eigvecs)`` of shapes (k,) and (p, k), in descending order.
    """
    xp, _ = get_namespace(A)
    if B is None:
        eigvals, eigvecs = xp.linalg.eigh(A)
    else:
        L = xp.linalg.cholesky(B)
        reduced = xp.linalg.solve(L, xp.linalg.solve(L, A).T)
        eigvals, u = xp.linalg.eigh((reduced + reduced.T) / 2)
        eigvecs = xp.linalg.solve(L.T, u)
    k = min(k, A.shape[0])
    return xp.flip(eigvals[-k:], axis=0), xp.flip(eigvecs[:, -k:], axis=1)


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
        deflated.append(
            view - score @ (score.T @ view) / norm_sq if norm_sq > 1e-12 else view
        )
    return deflated
