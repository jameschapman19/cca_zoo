"""Prototype: JIVEEY, the Eckart-Young loss of CCAEY with joint and individual columns.

CCAEY minimises the Eckart-Young loss of the generalized eigenproblem ``S w = lambda B w``,

    L(W) = -2 tr(W' S W) + tr((W' B W)^2),

where S is the block covariance of the views and ``B = blockdiag((1 - c) S_ii + c I)``. With
``Q = B^(1/2) W`` it is ``|| B^(-1/2) S B^(-1/2) - Q Q' ||_F^2`` up to a constant: the whitening is
inside the loss, through B in the quadratic term alone. JIVEEY is that same loss with W
restricted to the JIVE support: *joint* columns dense across views, *individual* columns
nonzero in one view only. B is block diagonal, so it keeps the support, and the loss is still
the exact whitened Frobenius fit of the covariance by joint + individual parts. Nothing is added:
the joint columns are rewarded for cross-view covariance, the individual columns for their own
view's variance, and the quadratic term penalises the overlap of a view's joint and individual
weights, which is JIVE's orthogonality. Scores are the projections ``X_i W_i``.

    L = -2 sum_ij <U_i, U_j> / D_ij + || sum_i [(1 - c) U_i'U_i / D_ii + c W_i'W_i] ||_F^2,
    U_i = X_i W_i,

with D_ij the samples views i and j share, less one, so views missing for some samples need
nothing more. With no individual columns it is CCAEY.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


class JIVEEY:
    """Joint and individual structure by the Eckart-Young loss, fitted by L-BFGS."""

    def __init__(
        self,
        n_joint: int,
        n_individual: int | list[int],
        shrinkage: float = 0.5,
        max_iter: int = 1000,
    ):
        self.n_joint, self.n_individual = n_joint, n_individual
        self.shrinkage, self.max_iter = shrinkage, max_iter

    # ----- data and parameters ------------------------------------------------
    def _prepare(self, views: list[np.ndarray]) -> None:
        """Centre each view on its observed samples and zero-fill the rest."""
        self.observed_ = np.stack([~np.isnan(v).any(axis=1) for v in views], axis=1)
        self.means_ = [np.nanmean(v, axis=0) for v in views]
        self.x_ = [np.nan_to_num(v - m) for v, m in zip(views, self.means_)]
        seen = self.observed_.astype(float)
        self.divisor_ = seen.T @ seen - 1.0
        self.p_ = [v.shape[1] for v in views]
        r = self.n_individual
        self.r_ = r if isinstance(r, list) else [r] * len(views)
        self.offset_ = self.n_joint + np.concatenate([[0], np.cumsum(self.r_)])

    def _factor(self, theta: np.ndarray) -> list[np.ndarray]:
        """Each view's weights W_i, joint columns first, then its own individual columns."""
        k, big_k = self.n_joint, self.n_joint + sum(self.r_)
        weights, at = [], 0
        for i, p in enumerate(self.p_):
            w = np.zeros((p, big_k))
            w[:, :k] = theta[at : at + p * k].reshape(p, k)
            at += p * k
            o = self.offset_[i]
            w[:, o : o + self.r_[i]] = theta[at : at + p * self.r_[i]].reshape(
                p, self.r_[i]
            )
            at += p * self.r_[i]
            weights.append(w)
        return weights

    def _flatten(self, per_view: list[np.ndarray]) -> np.ndarray:
        """The free entries of per-view matrices of W's shape, in ``_factor``'s order."""
        k = self.n_joint
        return np.concatenate(
            [
                np.concatenate(
                    [
                        m[:, :k].ravel(),
                        m[:, self.offset_[i] : self.offset_[i] + self.r_[i]].ravel(),
                    ]
                )
                for i, m in enumerate(per_view)
            ]
        )

    # ----- the loss ---------------------------------------------------------------
    def loss_and_gradient(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        """The Eckart-Young loss of the generalized eigenproblem, and its gradient."""
        c, inv = self.shrinkage, 1.0 / self.divisor_
        w = self._factor(theta)
        u = [x @ wi for x, wi in zip(self.x_, w)]
        m = len(w)
        gram = sum(
            (1 - c) * u[i].T @ u[i] * inv[i, i] + c * w[i].T @ w[i] for i in range(m)
        )
        reward = sum(
            np.sum(u[i] * u[j]) * inv[i, j] for i in range(m) for j in range(m)
        )
        loss = -2.0 * reward + np.sum(gram * gram)
        grad = [
            -4 * self.x_[i].T @ sum(inv[i, j] * u[j] for j in range(m))
            + 4 * ((1 - c) * self.x_[i].T @ u[i] * inv[i, i] + c * w[i]) @ gram
            for i in range(m)
        ]
        return float(loss), self._flatten(grad)

    # ----- fitting ----------------------------------------------------------------
    def fit(self, views: list[np.ndarray], seed: int = 0, n_init: int = 3) -> JIVEEY:
        """Fit from ``n_init`` random starts and keep the lowest loss."""
        self._prepare(views)
        rng = np.random.default_rng(seed)
        size = sum(p * (self.n_joint + r) for p, r in zip(self.p_, self.r_))
        scale = np.concatenate(
            [np.full(p * (self.n_joint + r), p**-0.5) for p, r in zip(self.p_, self.r_)]
        )
        best = None
        for _ in range(n_init):
            start = rng.standard_normal(size) * scale
            result = minimize(
                self.loss_and_gradient,
                start,
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": self.max_iter},
            )
            best = result if best is None or result.fun < best.fun else best
        self.weights_ = self._factor(best.x)
        self.loss_, self.n_iter_ = best.fun, best.nit
        return self

    # ----- scores -----------------------------------------------------------------
    def _projections(self, views: list[np.ndarray]) -> list[np.ndarray]:
        return [(v - m) @ w for v, m, w in zip(views, self.means_, self.weights_)]

    def transform(self, views: list[np.ndarray]) -> list[np.ndarray]:
        """Each view's joint scores, standardised; NaN rows for samples without the view."""
        out = []
        for z, o in zip(self._projections(views), self.observed_.T):
            joint = z[:, : self.n_joint]
            out.append((joint - joint[o].mean(axis=0)) / joint[o].std(axis=0))
        return out

    def transform_joint(self, views: list[np.ndarray]) -> np.ndarray:
        """One joint score per sample: the mean over the views that have it."""
        stacked = np.stack(self.transform(views))
        seen = np.isfinite(stacked)
        return np.where(seen, stacked, 0.0).sum(axis=0) / np.maximum(
            seen.sum(axis=0), 1
        )

    def transform_individual(self, views: list[np.ndarray]) -> list[np.ndarray]:
        """Each view's individual scores, standardised."""
        out = []
        for i, (z, o) in enumerate(zip(self._projections(views), self.observed_.T)):
            ind = z[:, self.offset_[i] : self.offset_[i] + self.r_[i]]
            out.append((ind - ind[o].mean(axis=0)) / ind[o].std(axis=0))
        return out

    # ----- variance explained -----------------------------------------------------
    def variance_explained(self, views: list[np.ndarray]) -> dict[str, np.ndarray]:
        """Each view's variance in its joint part, its individual part and the rest.

        On the samples that have the view, the centred data are regressed on the joint
        scores, then on the individual scores with the joint span removed, so the shares are
        additive: joint + individual + noise = 1. Components are taken in column order, each
        after the ones before it.

        Returns:
            ``joint`` of shape (n_views, n_joint) and ``individual`` of shape (n_views, r)
            with ``r`` the largest individual rank (NaN beyond a view's own), the per-view
            totals ``joint_total`` and ``individual_total``, and ``noise``.
        """
        n_views, r_max = len(views), max(self.r_)
        joint = np.full((n_views, self.n_joint), np.nan)
        individual = np.full((n_views, r_max), np.nan)
        for i, (z, v, m, o) in enumerate(
            zip(self._projections(views), views, self.means_, self.observed_.T)
        ):
            x, z = (v - m)[o], z[o]
            total = np.sum(x**2)
            q_joint = np.linalg.qr(z[:, : self.n_joint])[0]
            ind = z[:, self.offset_[i] : self.offset_[i] + self.r_[i]]
            q_ind = np.linalg.qr(ind - q_joint @ (q_joint.T @ ind))[0]
            joint[i] = np.sum((q_joint.T @ x) ** 2, axis=1) / total
            individual[i, : self.r_[i]] = np.sum((q_ind.T @ x) ** 2, axis=1) / total
        joint_total = np.nansum(joint, axis=1)
        individual_total = np.nansum(individual, axis=1)
        return {
            "joint": joint,
            "individual": individual,
            "joint_total": joint_total,
            "individual_total": individual_total,
            "noise": 1.0 - joint_total - individual_total,
        }
