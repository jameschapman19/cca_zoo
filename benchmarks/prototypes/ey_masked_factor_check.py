"""Check that the shared/private loss is the plain Eckart-Young loss on one masked factor.

Stack the shared loadings and every view's private loadings into one factor L of K = k +
sum(r_i) columns per view, the private columns of view j nonzero only in view j's rows. Then
the model covariance is L L', and the Frobenius loss is, block by block with weights w_ij and
divisors D_ij (U_i = X_i L_i, G_i = L_i' L_i, constants dropped):

    f = sum_ij w_ij ( -2 <U_i, U_j> / D_ij + <G_i, G_j> )
    grad L_i = sum_j w_ij ( -4 X_i' U_j / D_ij + 4 L_i G_j ),  masked to L's support

which is `-2 tr(L' S L) + tr((L' L)^2)` with its gradient `4 (L L'L - S L)`, restricted to a
support. The shared/private cross terms and the private terms of the other views fall out of
the Gram matrices G_i, with no case analysis.
    python ey_masked_factor_check.py
"""

from __future__ import annotations

import numpy as np
from ey_shared_private import SharedPrivateEY
from scipy.optimize import check_grad


def masked_factor_loss(x, divisor, weight, ranks, k, theta):
    """Loss and gradient on the stacked factor, from the same flat parameters as the class."""
    m, p = len(x), [v.shape[1] for v in x]
    big_k = k + sum(ranks)
    offset = np.concatenate([[0], np.cumsum(ranks)]) + k
    factor, at = [], 0
    for i in range(m):
        l_i = np.zeros((p[i], big_k))
        l_i[:, :k] = theta[at : at + p[i] * k].reshape(p[i], k)
        at += p[i] * k
        l_i[:, offset[i] : offset[i] + ranks[i]] = theta[
            at : at + p[i] * ranks[i]
        ].reshape(p[i], ranks[i])
        at += p[i] * ranks[i]
        factor.append(l_i)
    u = [xi @ li for xi, li in zip(x, factor)]
    g = [li.T @ li for li in factor]
    coef = weight / divisor
    loss = sum(
        weight[i, j] * np.sum(g[i] * g[j]) - 2 * coef[i, j] * np.sum(u[i] * u[j])
        for i in range(m)
        for j in range(m)
    )
    grad = []
    for i in range(m):
        full = sum(
            -4 * coef[i, j] * x[i].T @ u[j] + 4 * weight[i, j] * factor[i] @ g[j]
            for j in range(m)
        )
        grad.append(
            np.concatenate(
                [full[:, :k].ravel(), full[:, offset[i] : offset[i] + ranks[i]].ravel()]
            )
        )
    return float(loss), np.concatenate(grad)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    views = [rng.standard_normal((50, p)) + 0.5 for p in (8, 6, 5)]
    views[0][:10], views[2][40:] = np.nan, np.nan
    for weighting in ("equal", "count"):
        model = SharedPrivateEY(2, [2, 1, 1], weighting=weighting)
        model._prepare(views)
        model.p_, model.r_ = [8, 6, 5], [2, 1, 1]
        theta = (
            rng.standard_normal(sum(p * (2 + r) for p, r in zip(model.p_, model.r_)))
            * 0.3
        )
        old = model.loss_and_gradient(theta)
        new = masked_factor_loss(
            model.x_, model.divisor_, model.weight_, model.r_, 2, theta
        )
        print(
            f"{weighting:>6}: loss {old[0]:.10f} vs {new[0]:.10f}; "
            f"max gradient difference {np.max(np.abs(old[1] - new[1])):.2e}; "
            f"finite-difference error {check_grad(lambda t: masked_factor_loss(model.x_, model.divisor_, model.weight_, model.r_, 2, t)[0], lambda t: masked_factor_loss(model.x_, model.divisor_, model.weight_, model.r_, 2, t)[1], theta) / np.linalg.norm(new[1]):.1e}"
        )
