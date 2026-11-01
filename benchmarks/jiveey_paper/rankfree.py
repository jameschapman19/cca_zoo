"""JIVEEY with the ranks as upper bounds and a group penalty that prunes columns.

The Eckart-Young loss gives each column q_k of Q = C^(1/2) W the squared norm G_kk, which at a
minimum is an eigenvalue of the whitened covariance; sample noise leaves small positive ones.
Adding ``penalty * sum_k sqrt(G_kk)`` (a group lasso over whole columns, smoothed by ``eps``)
shrinks those towards zero, so the rank is the number of columns that survive.
"""

from __future__ import annotations

import numpy as np
from common import JIVEEY


class RankFreeJIVEEY(JIVEEY):
    """JIVEEY with a smoothed group-lasso penalty on the columns of Q."""

    def __init__(self, *args, penalty: float, eps: float = 1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.penalty, self.eps = penalty, eps

    def loss_and_gradient(self, theta):
        c, inv = self.shrinkage, 1.0 / self.divisor_
        w = self._factor(theta)
        u = [x @ wi for x, wi in zip(self.x_, w)]
        m = len(w)
        within = [
            (1 - c) * self.x_[i].T @ u[i] * inv[i, i] + c * w[i] for i in range(m)
        ]  # C_i W_i
        gram = sum(w[i].T @ within[i] for i in range(m))
        reward = sum(np.sum(u[i] * u[j]) * inv[i, j] for i in range(m) for j in range(m))
        root = np.sqrt(np.diag(gram) + self.eps)
        loss = -2.0 * reward + np.sum(gram * gram) + self.penalty * np.sum(root)
        grad = [
            -4 * self.x_[i].T @ sum(inv[i, j] * u[j] for j in range(m))
            + 4 * within[i] @ gram
            + self.penalty * within[i] / root
            for i in range(m)
        ]
        return float(loss), self._flatten(grad)

    def column_strengths(self):
        """Each column's G_kk: joint columns first, then each view's individual columns."""
        c = self.shrinkage
        m = len(self.weights_)
        inv = 1.0 / self.divisor_
        gram = sum(
            w.T @ ((1 - c) * x.T @ (x @ w) * inv[i, i] + c * w)
            for i, (x, w) in enumerate(zip(self.x_, self.weights_))
        )
        return np.diag(gram)
