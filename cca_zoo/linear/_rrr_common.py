"""Shared helpers for reduced-rank-regression CCA methods (CCAR3, ECCA)."""

from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf


def _sqrt_inv_psd(S: np.ndarray, threshold: float = 1e-4) -> np.ndarray:
    """Symmetric inverse square root of a PSD matrix, zeroing small eigenvalues."""
    vals, vecs = np.linalg.eigh(S)
    inv_sqrt_vals = np.where(vals > threshold, 1.0 / np.sqrt(np.abs(vals)), 0.0)
    return (vecs * inv_sqrt_vals) @ vecs.T


def _whiten_factor(G: np.ndarray, ridge: float) -> np.ndarray:
    """Return W such that W.T @ G @ W == I, via a (jittered) Cholesky factor."""
    p = G.shape[0]
    G = (G + G.T) / 2 + ridge * np.eye(p)
    try:
        L = np.linalg.cholesky(G)
        return np.asarray(np.linalg.inv(L).T)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(G)
        vals = np.maximum(vals, ridge)
        return np.asarray((vecs * (1.0 / np.sqrt(vals))) @ vecs.T)


def _whiten_response(Y: np.ndarray, ledoit_wolf: bool) -> tuple[np.ndarray, np.ndarray]:
    """Whiten the response view by its (optionally Ledoit-Wolf shrunk) covariance.

    Returns ``(Y_tilde, sqrt_inv_Sy)`` where ``Y_tilde = Y @ sqrt_inv_Sy``.
    """
    n = Y.shape[0]
    Sy = LedoitWolf().fit(Y).covariance_ if ledoit_wolf else Y.T @ Y / n
    sqrt_inv_Sy = _sqrt_inv_psd(Sy)
    return Y @ sqrt_inv_Sy, sqrt_inv_Sy


def _postprocess_rrr_fit(
    B: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    sqrt_inv_Sy: np.ndarray,
    r: int,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn a reduced-rank coefficient matrix into whitened canonical directions."""
    n, p = X.shape
    q = Y.shape[1]

    if not np.any(B):
        return np.zeros((p, r)), np.zeros((q, r))

    r_eff = min(r, *B.shape)
    U0, _, Vt0 = np.linalg.svd(B, full_matrices=False)
    U0 = U0[:, :r_eff]
    V0 = sqrt_inv_Sy @ Vt0[:r_eff, :].T

    XU0 = X @ U0
    YV0 = Y @ V0
    GX = XU0.T @ XU0 / n
    GY = YV0.T @ YV0 / n

    U = U0 @ _whiten_factor(GX, ridge)
    V = V0 @ _whiten_factor(GY, ridge)

    XU = X @ U
    YV = Y @ V
    cor = np.diag(XU.T @ YV / n).copy()

    neg = cor < 0
    V[:, neg] *= -1
    cor[neg] *= -1

    order = np.argsort(-cor)
    U, V, cor = U[:, order], V[:, order], cor[order]

    if r_eff < r:
        U = np.hstack([U, np.zeros((p, r - r_eff))])
        V = np.hstack([V, np.zeros((q, r - r_eff))])
    return U, V
