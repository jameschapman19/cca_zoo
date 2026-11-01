"""Prototype: an Eckart-Young shared/private decomposition.

The block covariance of the views is fitted, in squared Frobenius error, by

    L_S L_S'  +  blockdiag(L_P1 L_P1', ..., L_PM L_PM'),

a rank-k shared part whose loadings are stacked across views, and a low-rank
private part per view that touches only that view's diagonal block. Unconstrained
and differentiable, as the Eckart-Young loss of CCAEY is. Every term is formed from
the n by k products X_i L, never a feature-by-feature matrix, and a sample that lacks
a view drops out of the blocks it is not in, with the divisor of the samples that
have both views, as CCAEY's missing-view loss does.

Not a library model: this is to see whether the idea works.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


class SharedPrivateEY:
    """Fit the loss above by L-BFGS from an initial shared latent."""

    def __init__(
        self,
        n_shared: int,
        n_private: int | list[int],
        max_iter: int = 500,
        weighting: str = "equal",
    ):
        self.n_shared, self.n_private, self.max_iter = n_shared, n_private, max_iter
        self.weighting = weighting

    # ----- data ------------------------------------------------------------
    def _prepare(self, views: list[np.ndarray]) -> None:
        """Centre on observed rows, zero-fill, scale each view to unit total variance."""
        self.observed_ = np.stack([~np.isnan(v).any(axis=1) for v in views], axis=1)
        self.means_ = [np.nanmean(v, axis=0) for v in views]
        x = [np.nan_to_num(v - m) for v, m in zip(views, self.means_)]
        self.scales_ = [
            np.sqrt(np.sum(v**2) / (o.sum() - 1)) for v, o in zip(x, self.observed_.T)
        ]
        self.x_ = [v / s for v, s in zip(x, self.scales_)]
        seen = self.observed_.astype(float)
        self.divisor_ = seen.T @ seen - 1.0
        # Each block of the covariance is estimated from the samples that have both its
        # views, so a block from few samples is noisier: "count" weights blocks by that number.
        self.weight_ = (
            self.divisor_ / self.divisor_.max()
            if self.weighting == "count"
            else np.ones_like(self.divisor_)
        )

    # ----- loss and gradient -------------------------------------------------
    def _unpack(self, theta: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        shared, private, at = [], [], 0
        for p, r in zip(self.p_, self.r_):
            shared.append(theta[at : at + p * self.n_shared].reshape(p, self.n_shared))
            at += p * self.n_shared
            private.append(theta[at : at + p * r].reshape(p, r))
            at += p * r
        return shared, private

    def loss_and_gradient(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        """The weighted Frobenius loss up to a constant, and its gradient."""
        shared, private = self._unpack(theta)
        m, inv, w = len(shared), 1.0 / self.divisor_, self.weight_
        u = [x @ s for x, s in zip(self.x_, shared)]
        v = [x @ q for x, q in zip(self.x_, private)]
        gram = [s.T @ s for s in shared]
        weighted_gram = [sum(w[i, j] * gram[j] for j in range(m)) for i in range(m)]
        loss = sum(np.sum(gram[i] * weighted_gram[i]) for i in range(m))
        grad_s, grad_p = [], []
        for i in range(m):
            cross = shared[i].T @ private[i]
            h = private[i].T @ private[i]
            loss += w[i, i] * (np.sum(h * h) + 2 * np.sum(cross**2))
            loss -= 2 * w[i, i] * np.sum(v[i] ** 2) * inv[i, i]
            for j in range(m):
                loss -= 2 * w[i, j] * np.sum(u[i] * u[j]) * inv[i, j]
            others = sum(w[i, j] * inv[i, j] * u[j] for j in range(m))
            grad_s.append(
                -4 * self.x_[i].T @ others
                + 4 * shared[i] @ weighted_gram[i]
                + 4 * w[i, i] * private[i] @ cross.T
            )
            grad_p.append(
                -4 * w[i, i] * inv[i, i] * self.x_[i].T @ v[i]
                + 4 * w[i, i] * (private[i] @ h + shared[i] @ cross)
            )
        flat = np.concatenate(
            [np.concatenate([a.ravel(), b.ravel()]) for a, b in zip(grad_s, grad_p)]
        )
        return float(loss), flat

    # ----- fitting -----------------------------------------------------------
    def fit(
        self, views: list[np.ndarray], latent: np.ndarray | None = None, seed: int = 0
    ) -> SharedPrivateEY:
        """Fit; ``latent`` (n, k) starts the shared part, else a random start."""
        self._prepare(views)
        self.p_ = [v.shape[1] for v in views]
        r = self.n_private
        self.r_ = r if isinstance(r, list) else [r] * len(views)
        rng = np.random.default_rng(seed)
        theta = []
        for i, x in enumerate(self.x_):
            rows = self.observed_[:, i]
            if latent is None:
                shared = rng.standard_normal((x.shape[1], self.n_shared)) * 0.05
            else:
                z = np.where(np.isfinite(latent), latent, 0.0)
                shared = x[rows].T @ z[rows] / (rows.sum() - 1)
            resid = (
                x - np.where(rows[:, None], z @ shared.T, 0.0)
                if latent is not None
                else x
            )
            _, s, vt = np.linalg.svd(resid[rows], full_matrices=False)
            private = vt[: self.r_[i]].T * s[: self.r_[i]] / np.sqrt(rows.sum() - 1)
            theta.append(np.concatenate([shared.ravel(), private.ravel()]))
        result = minimize(
            self.loss_and_gradient,
            np.concatenate(theta),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter},
        )
        self.shared_, self.private_ = self._unpack(result.x)
        self.loss_, self.n_iter_ = result.fun, result.nit
        return self

    # ----- scores ------------------------------------------------------------
    def scores(self, views: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
        """Shared scores (n, k) averaged over each sample's views, and private scores.

        Each view's centred rows are regressed on its shared and private loadings
        together, so that private variation does not leak into the shared scores.
        """
        shared_scores, private_scores = [], []
        for v, m, s, shared, private in zip(
            views, self.means_, self.scales_, self.shared_, self.private_
        ):
            loadings = np.concatenate([shared, private], axis=1)
            z = ((v - m) / s) @ np.linalg.pinv(loadings).T
            shared_scores.append(z[:, : self.n_shared])
            private_scores.append(z[:, self.n_shared :])
        stacked = np.stack(shared_scores)
        seen = np.isfinite(stacked)
        total = np.where(seen, stacked, 0.0).sum(axis=0)
        return total / np.maximum(seen.sum(axis=0), 1), private_scores


class WhitenedSharedPrivateEY:
    """The same fit on ridge-whitened views, as CCAEY's ``shrinkage`` whitens its constraint.

    Each view is mapped to ``(x - mean) V diag(d)^(-1/2)``, with ``x = U S V'`` its SVD and
    ``d = (1 - c) S^2 / (n - 1) + c`` the spectrum of ``(1 - c) Sigma + c I``. The Frobenius fit
    of the block covariance of these coordinates is then the Eckart-Young loss on the whitened
    covariance: ``c = 1`` is the raw-covariance fit (PLS, which chases large private variance),
    and as ``c`` falls the large directions are compressed, as CCA does. The coordinates have at
    most ``n - 1`` dimensions however wide the view, so wide views cost the SVD alone.
    """

    def __init__(
        self,
        n_shared: int,
        n_private: int | list[int],
        shrinkage: float = 0.5,
        max_iter: int = 500,
        weighting: str = "equal",
    ):
        self.n_shared, self.n_private = n_shared, n_private
        self.shrinkage, self.max_iter, self.weighting = shrinkage, max_iter, weighting

    def _reduce(self, views: list[np.ndarray]) -> list[np.ndarray]:
        reduced = []
        for v, mean, basis in zip(views, self.means_, self.bases_):
            y = (v - mean) @ basis
            reduced.append(y)
        return reduced

    def fit(self, views: list[np.ndarray], seed: int = 0) -> WhitenedSharedPrivateEY:
        """Whiten each view on its observed samples, then fit the shared/private loss."""
        c = self.shrinkage
        self.means_, self.bases_ = [], []
        for v in views:
            rows = ~np.isnan(v).any(axis=1)
            mean = v[rows].mean(axis=0)
            x = v[rows] - mean
            _, s, vt = np.linalg.svd(x, full_matrices=False)
            keep = s > s[0] * max(x.shape) * np.finfo(s.dtype).eps
            d = (1.0 - c) * s[keep] ** 2 / (rows.sum() - 1) + c
            self.means_.append(mean)
            self.bases_.append(vt[keep].T / np.sqrt(d))
        self.inner_ = SharedPrivateEY(
            self.n_shared, self.n_private, self.max_iter, self.weighting
        ).fit(self._reduce(views), seed=seed)
        return self

    def scores(self, views: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
        """Shared scores (n, k) and private scores, from the whitened coordinates."""
        return self.inner_.scores(self._reduce(views))
