r"""Shared machinery for the Eckart-Young (EY) unconstrained CCA objective.

This is the loss used by :class:`~cca_zoo.linear.gradient.CCA_EY`,
:class:`~cca_zoo.linear.gradient.MCCA_EY`, :class:`~cca_zoo.deep.DCCA_EY`, and
:class:`~cca_zoo.tree.TreeCCA`: an unconstrained (no manifold projection
required) stand-in for canonical correlation analysis that is a stationary
point exactly at the canonical directions.

For $M$ views with (possibly non-orthonormal) embeddings
$Z_1, \dots, Z_M$, each $(n, k)$, define the mean pairwise
cross-covariance and mean auto-covariance:

$$
C = \frac{1}{M} \sum_{i, j} \operatorname{Cov}(Z_i, Z_j), \qquad
V = \frac{1}{M} \sum_i \operatorname{Cov}(Z_i, Z_i)
$$

(the sum for $C$ ranges over *all* ordered pairs, including
$i = j$). The EY loss to minimise is:

$$
\mathcal{L}_{EY} = -2 \operatorname{tr}(C) + \operatorname{tr}(V V)
$$

References:
    Chapman, J., Wells, L., & Lawry Aguila, A. (2024). Unconstrained
    Stochastic CCA: Unifying Multiview and Self-Supervised Learning.
    arXiv:2310.01012.
"""

from __future__ import annotations

import numpy as np


def ey_cross_covariance(
    representations: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    r"""Mean pairwise cross-covariance and mean auto-covariance of M embeddings.

    Args:
        representations: List of M arrays, each of shape (n_samples, k).
            Need not be pre-centred; centring is performed internally.

    Returns:
        Tuple ``(C, V)``, each of shape (k, k):
        ``C`` is the mean of $\operatorname{Cov}(Z_i, Z_j)$ over all
        ordered pairs (including $i = j$); ``V`` is the mean of
        $\operatorname{Cov}(Z_i, Z_i)$.
    """
    n = representations[0].shape[0]
    m = len(representations)
    centred = [z - z.mean(axis=0) for z in representations]
    k = centred[0].shape[1]
    C = np.zeros((k, k))
    V = np.zeros((k, k))
    for zi in centred:
        V += zi.T @ zi / (n - 1)
        for zj in centred:
            C += zi.T @ zj / (n - 1)
    return C / m, V / m


def ey_loss(representations: list[np.ndarray]) -> dict[str, float]:
    """Compute the EY loss and its reward/penalty components.

    Args:
        representations: List of M arrays, each of shape (n_samples, k).

    Returns:
        Dictionary with ``"objective"`` (``-rewards + penalties``, to be
        minimised), ``"rewards"`` (``2 * tr(C)``), and ``"penalties"``
        (``tr(V @ V)``).
    """
    C, V = ey_cross_covariance(representations)
    rewards = float(np.trace(2.0 * C))
    penalties = float(np.trace(V @ V))
    return {
        "objective": -rewards + penalties,
        "rewards": rewards,
        "penalties": penalties,
    }


def weight_gram_mean(weights: list[np.ndarray]) -> np.ndarray:
    r"""Mean weight Gram matrix $B = \frac{1}{M} \sum_i W_i^\top W_i$.

    Args:
        weights: Per-view weight matrices, each of shape (p_i, k).

    Returns:
        Matrix of shape (k, k).
    """
    total: np.ndarray = sum(w.T @ w for w in weights) / len(weights)
    return total


def random_orthonormal_weights(
    views: list[np.ndarray], latent_dimensions: int, rng: np.random.Generator
) -> list[np.ndarray]:
    r"""Cheap, data-independent initial weights with orthonormal columns.

    Each view's weight matrix is the $Q$ factor of a QR decomposition of
    an i.i.d. standard normal matrix, so $W_i^\top W_i = I$ exactly, before
    any gradient step and without looking at the data at all. This matches
    the structure of :class:`~cca_zoo.linear.gradient.PLS_EY`'s own penalty,
    which drives weights towards orthonormality directly in weight space
    (see :func:`weight_gram_mean`) — the loss's own fixed point is already
    the natural initial point's shape.

    Args:
        views: Per-view arrays; only ``.shape[1]`` (feature count) is used.
        latent_dimensions: Requested number of latent dimensions.
        rng: Random generator.

    Returns:
        List of weight matrices, each $(p_i, k)$ with orthonormal columns,
        where $k = \min(\text{latent\_dimensions}, p_i)$.
    """
    weights = []
    for v in views:
        p = v.shape[1]
        k = min(latent_dimensions, p)
        w, _ = np.linalg.qr(rng.standard_normal((p, k)))
        weights.append(w)
    return weights


def cheap_orthonormal_projection_weights(
    views: list[np.ndarray],
    latent_dimensions: int,
    batch_size: int | None,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    r"""Cheap initial weights giving unit-variance projections on one batch.

    Classical CCA whitens each view with a full $(p, p)$ eigendecomposition
    of its covariance before fitting; that full-batch pass is exactly what
    the EY reformulation exists to avoid (see
    :class:`~cca_zoo.linear.gradient.CCA_EY`). This is a cheap substitute
    usable only at initialisation: draw random directions, project one
    mini-batch, and QR-orthonormalise the resulting $(n, k)$ projection
    instead of the $(p, p)$ data covariance, then pull that
    orthonormalisation back into the weight matrix via the QR's $(k, k)$
    triangular factor — one small QR and one $k \times k$ solve per view,
    independent of $p$. Concretely, for random directions $W_0$,
    mini-batch $X$, and $X W_0 = QR$:

    $$
    W = W_0 R^{-1}, \qquad X W = X W_0 R^{-1} = Q R R^{-1} = Q
    $$

    so the resulting projections are exactly orthonormal ($Q^\top Q = I$)
    on that mini-batch — a cheap stand-in for the reward term's ideal
    starting point ($V \approx I$; see :func:`ey_cross_covariance`) and
    the natural match for :class:`~cca_zoo.linear.gradient.CCA_EY`'s own
    fixed point. This orthonormalises only the *first* mini-batch, though:
    every later step draws an independent fresh batch, so this does not,
    by itself, prevent the ``c=0`` divergence risk noted in that class's
    docstring (empirically confirmed: it does not measurably postpone it
    either) — ``c`` or ``batch_size`` remain the actual remedy for that.

    Args:
        views: Per-view arrays, each $(n, p_i)$.
        latent_dimensions: Requested number of latent dimensions.
        batch_size: Mini-batch size used for the initial projection.
            ``None`` uses the full dataset.
        rng: Random generator.

    Returns:
        List of weight matrices, each $(p_i, k)$.
    """
    n = views[0].shape[0]
    bs = n if batch_size is None else min(batch_size, n)
    idx = rng.choice(n, bs, replace=False)
    weights = []
    for v in views:
        p = v.shape[1]
        k = min(latent_dimensions, p)
        w0, _ = np.linalg.qr(rng.standard_normal((p, k)))
        z0 = v[idx] @ w0
        _, r = np.linalg.qr(z0)
        weights.append(w0 @ np.linalg.solve(r, np.eye(k)))
    return weights


def ey_grad_z(representations: list[np.ndarray]) -> list[np.ndarray]:
    r"""Gradient of the EY loss w.r.t. each embedding (M-view generalised).

    $$
    \frac{\partial \mathcal{L}_{EY}}{\partial Z_k}
        = \frac{4}{M (n - 1)} \left( \tilde{Z}_k V - S \right)
    $$

    where $\tilde{Z}_k$ is the centred k-th embedding,
    $S = \sum_i \tilde{Z}_i$, and $V$ is the mean auto-covariance
    (see :func:`ey_cross_covariance`). Verified against finite-difference
    gradients of :func:`ey_loss` for M = 2, 3, 4.

    Args:
        representations: List of M arrays, each of shape (n_samples, k).

    Returns:
        List of M gradient arrays, each of shape (n_samples, k), one per view.
    """
    n = representations[0].shape[0]
    m = len(representations)
    centred = [z - z.mean(axis=0) for z in representations]
    total = sum(centred)
    _, V = ey_cross_covariance(representations)
    scale = 4.0 / (m * (n - 1))
    return [scale * (zc @ V - total) for zc in centred]
