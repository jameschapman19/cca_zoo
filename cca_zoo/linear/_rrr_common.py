"""Shared helpers for reduced-rank-regression CCA methods (CCAR3, ECCA)."""

from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf


def _sqrt_inv_psd(S: np.ndarray, threshold: float = 1e-4) -> np.ndarray:
    """Symmetric inverse square root of a PSD matrix.

    Eigenvalues below ``threshold`` times the largest are zeroed, so the
    result does not depend on the units of the data.
    """
    vals, vecs = np.linalg.eigh(S)
    keep = vals > threshold * vals.max()
    inv_sqrt_vals = np.where(keep, 1.0 / np.sqrt(np.abs(vals)), 0.0)
    return (vecs * inv_sqrt_vals) @ vecs.T


def _whiten_factor(G: np.ndarray, ridge: float) -> np.ndarray:
    """Return W such that W.T @ G @ W == I for the ridge-regularised G."""
    vals, vecs = np.linalg.eigh((G + G.T) / 2 + ridge * np.eye(G.shape[0]))
    return np.asarray((vecs / np.sqrt(np.maximum(vals, ridge))) @ vecs.T)


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
    """Canonical directions from the regression of the whitened ``Y`` on ``X``.

    ``B`` maps ``X`` to ``Y @ sqrt_inv_Sy``. The canonical correlations are the
    singular values of the fitted values ``X @ B``, not of ``B``, whose
    singular vectors ignore the covariance of ``X``.
    """
    n, p = X.shape
    q = Y.shape[1]

    if not np.any(B):
        return np.zeros((p, r)), np.zeros((q, r))

    r_eff = min(r, *B.shape)
    _, _, Qt = np.linalg.svd(X @ B, full_matrices=False)
    Q = Qt[:r_eff].T
    U0 = B @ Q
    V0 = sqrt_inv_Sy @ Q

    XU0 = X @ U0
    YV0 = Y @ V0
    GX = XU0.T @ XU0 / (n - 1)
    GY = YV0.T @ YV0 / (n - 1)

    U = U0 @ _whiten_factor(GX, ridge)
    V = V0 @ _whiten_factor(GY, ridge)
    # Whitening each side separately leaves the pairs mixed; the SVD of the
    # whitened cross-covariance gives the canonical pairs, in order of their
    # correlations, all positive.
    left, _, right_t = np.linalg.svd((X @ U).T @ (Y @ V) / n)
    U, V = U @ left, V @ right_t.T

    if r_eff < r:
        U = np.hstack([U, np.zeros((p, r - r_eff))])
        V = np.hstack([V, np.zeros((q, r - r_eff))])
    return U, V
