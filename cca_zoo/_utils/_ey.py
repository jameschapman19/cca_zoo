r"""Shared machinery for the Eckart-Young (EY) unconstrained CCA objective.

This is the loss used by :class:`~cca_zoo.linear.gradient.CCA_EY`,
:class:`~cca_zoo.linear.gradient.MCCA_EY`, :class:`~cca_zoo.deep.DCCA_EY`, and
:class:`~cca_zoo.tree.TreeCCA`: an unconstrained (no manifold projection
required) stand-in for canonical correlation analysis that is a stationary
point exactly at the canonical directions.

For :math:`M` views with (possibly non-orthonormal) embeddings
:math:`Z_1, \dots, Z_M`, each :math:`(n, k)`, define the mean pairwise
cross-covariance and mean auto-covariance:

.. math::

    C = \frac{1}{M} \sum_{i, j} \operatorname{Cov}(Z_i, Z_j), \qquad
    V = \frac{1}{M} \sum_i \operatorname{Cov}(Z_i, Z_i)

(the sum for :math:`C` ranges over *all* ordered pairs, including
:math:`i = j`). The EY loss to minimise is:

.. math::

    \mathcal{L}_{EY} = -2 \operatorname{tr}(C) + \operatorname{tr}(V V)

Reference:
    Chapman, J., Lawry Aguila, A., & Wells, L. (2022). A Generalised
    EigenGame with Extensions to Multiview Representation Learning.
    arXiv:2211.11323.
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
        ``C`` is the mean of :math:`\operatorname{Cov}(Z_i, Z_j)` over all
        ordered pairs (including :math:`i = j`); ``V`` is the mean of
        :math:`\operatorname{Cov}(Z_i, Z_i)`.
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


def ey_grad_z(representations: list[np.ndarray]) -> list[np.ndarray]:
    r"""Gradient of the EY loss w.r.t. each embedding (M-view generalised).

    .. math::

        \frac{\partial \mathcal{L}_{EY}}{\partial Z_k}
            = \frac{4}{M (n - 1)} \left( \tilde{Z}_k V - S \right)

    where :math:`\tilde{Z}_k` is the centred k-th embedding,
    :math:`S = \sum_i \tilde{Z}_i`, and :math:`V` is the mean auto-covariance
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
