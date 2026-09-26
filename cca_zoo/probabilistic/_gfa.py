"""Group Factor Analysis, ported from the R package CCAGFA."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from cca_zoo._base import BaseModel
from cca_zoo.probabilistic._utils import PosteriorMeanTransformMixin

# CCAGFA::getDefaultOpts() priors: near-improper/flat, matching the R
# package's defaults exactly (prior.alpha_0/beta_0/alpha_0t/beta_0t <- 1e-14).
_ARD_ALPHA_0 = 1e-14
_ARD_BETA_0 = 1e-14
_TAU_ALPHA_0 = 1e-14
_TAU_BETA_0 = 1e-14
_INIT_TAU = 1e3
_DROP_TOL = 1e-7
_PATIENCE = 1000


class GFA(PosteriorMeanTransformMixin, BaseModel):
    r"""Group Factor Analysis: Bayesian CCA with per-view ARD.

    A port of ``GFA()`` from the R package CCAGFA. A shared latent $z$
    generates every view, with an ARD precision per view and dimension:

    $$
    \begin{aligned}
    \alpha_{i,k} &\sim \mathrm{Gamma}(a_0, b_0), &
    W_i[:, k] &\sim \mathcal{N}(0, \alpha_{i,k}^{-1} I), \\
    z &\sim \mathcal{N}(0, I), &
    x_i \mid z &\sim \mathcal{N}(W_i z, \tau_i^{-1} I),
    \end{aligned}
    $$

    so a dimension is shared when $\alpha_{i,k}$ is small in several views
    and private when small in one. Inference is closed-form coordinate-ascent
    variational Bayes. ``n_components`` is an upper bound: dimensions with
    vanishing loadings are pruned, leaving ``n_components_``. Convergence is
    judged on the change in $z$ rather than the lower bound, so raise
    ``max_iter`` if ``n_components_`` is larger than expected.

    Args:
        n_components: Upper bound on the number of latent dimensions.
            Default is 1.
        center: Whether to centre each view. Default is True.
        max_iter: Maximum coordinate-ascent iterations. Default is 10000.
        tol: Tolerance on the relative change in $z$, held for 1000
            iterations. Default is 1e-4.
        drop_k: Whether to prune unused dimensions, as CCAGFA's ``dropK``.
            Default is True.
        n_posterior_samples: Draws from the fitted posterior. Default is 1000.
        random_state: Seed for the initialisation. Default is None.

    Attributes:
        weights_: Posterior mean loadings of each view, shape
            (n_features_i, n_components_).
        view_relevance_: Posterior mean ARD precisions, shape
            (n_views, n_components_); large means shrunk away.
        n_components_: Number of dimensions kept.
        posterior_samples_: Posterior draws keyed ``W_{i}``, ``log_psi_{i}``
            and ``alpha``.
        n_iter_: Number of iterations run.

    References:
        Klami, A., Virtanen, S., & Kaski, S. (2013). Bayesian Canonical
        Correlation Analysis. Journal of Machine Learning Research, 14,
        965-1003.

    Example:
        >>> import numpy as np
        >>> from cca_zoo.probabilistic import GFA
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 4))
        >>> X2 = rng.standard_normal((50, 3))
        >>> model = GFA(n_components=2, max_iter=50).fit([X1, X2])
    """

    def __init__(
        self,
        n_components: int = 1,
        center: bool = True,
        max_iter: int = 10000,
        tol: float = 1e-4,
        drop_k: bool = True,
        n_posterior_samples: int = 1000,
        random_state: int | None = None,
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.max_iter = max_iter
        self.tol = tol
        self.drop_k = drop_k
        self.n_posterior_samples = n_posterior_samples
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Public fit
    # ------------------------------------------------------------------

    def fit(self, views: list[ArrayLike], y: None = None) -> GFA:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.

        Returns:
            self.
        """
        validated = self._setup_fit(views)
        rng = np.random.default_rng(self.random_state)

        views_arr = [np.asarray(v, dtype=float) for v in validated]
        m_views = len(views_arr)
        n = views_arr[0].shape[0]
        d = [v.shape[1] for v in views_arr]
        k = self.n_components

        # --- initialization (CCAGFA::GFA(), matching getDefaultOpts()) ---
        z = rng.standard_normal((n, k))
        cov_z = np.eye(k)
        w = [np.zeros((d[m], k)) for m in range(m_views)]
        cov_w = [np.eye(k) for m in range(m_views)]
        tau = np.full(m_views, _INIT_TAU)
        datavar = np.array(
            [np.var(views_arr[m], axis=0, ddof=1).sum() for m in range(m_views)]
        )
        alpha = [
            np.full(k, k * d[m] / max(datavar[m] - 1.0 / tau[m], 1e-8))
            for m in range(m_views)
        ]

        y_const = np.array([np.sum(views_arr[m] ** 2) for m in range(m_views)])
        a_ard = _ARD_ALPHA_0 + np.array(d) / 2.0  # (M,), constant across iters
        a_tau = _TAU_ALPHA_0 + n * np.array(d) / 2.0  # (M,), constant

        ww = [w[m].T @ w[m] + d[m] * cov_w[m] for m in range(m_views)]
        zz = z.T @ z + n * cov_z
        b_ard = [np.full(k, _ARD_BETA_0) for _ in range(m_views)]
        b_tau = np.full(m_views, _TAU_BETA_0)

        # Requires `_PATIENCE` consecutive iterations with small relative
        # change AND no pruning event, not just one: rel_change can dip
        # below tol for a few hundred iterations in the middle of a slow
        # ARD-driven pruning process (one dimension's decay temporarily
        # dominated by a faster-settling one finishing first) before
        # rising again once that's the only signal left. A single-iteration
        # check was caught declaring convergence during exactly such a lull,
        # 2500+ iterations before the pruning it was still waiting on.
        prev_z: np.ndarray | None = None
        n_iter = self.max_iter
        stable_count = 0
        for iteration in range(self.max_iter):
            # --- W update (per view), using the current zz ---
            for m in range(m_views):
                tmp = 1.0 / np.sqrt(alpha[m])
                inner = np.outer(tmp, tmp) * zz + np.eye(k) / tau[m]
                cho_w = np.linalg.cholesky(inner)
                inv_inner = np.linalg.solve(cho_w.T, np.linalg.solve(cho_w, np.eye(k)))
                cov_w[m] = (1.0 / tau[m]) * np.outer(tmp, tmp) * inv_inner
                w[m] = views_arr[m].T @ z @ cov_w[m] * tau[m]
                ww[m] = w[m].T @ w[m] + d[m] * cov_w[m]

            # --- Z update, using the just-updated W ---
            precision_z = np.eye(k)
            for m in range(m_views):
                precision_z = precision_z + tau[m] * ww[m]
            cho_z = np.linalg.cholesky(precision_z)
            cov_z = np.linalg.solve(cho_z.T, np.linalg.solve(cho_z, np.eye(k)))
            rhs = np.zeros((n, k))
            for m in range(m_views):
                rhs = rhs + views_arr[m] @ w[m] * tau[m]
            z = rhs @ cov_z
            zz = z.T @ z + n * cov_z

            # --- alpha update (per view), using the just-updated ww ---
            for m in range(m_views):
                b_ard[m] = _ARD_BETA_0 + np.diag(ww[m]) / 2.0
                alpha[m] = a_ard[m] / b_ard[m]

            # --- tau update (per view) ---
            for m in range(m_views):
                b_tau[m] = (
                    _TAU_BETA_0
                    + (
                        y_const[m]
                        + np.sum(ww[m] * zz)
                        - 2.0 * np.sum(z * (views_arr[m] @ w[m]))
                    )
                    / 2.0
                )
                tau[m] = a_tau[m] / b_tau[m]

            # --- dynamic component pruning (dropK) ---
            pruned = False
            if self.drop_k:
                keep = np.where(np.mean(z**2, axis=0) > _DROP_TOL)[0]
                if 0 < len(keep) != k:
                    pruned = True
                    k = len(keep)
                    z = z[:, keep]
                    cov_z = cov_z[np.ix_(keep, keep)]
                    zz = zz[np.ix_(keep, keep)]
                    for m in range(m_views):
                        w[m] = w[m][:, keep]
                        cov_w[m] = cov_w[m][np.ix_(keep, keep)]
                        ww[m] = ww[m][np.ix_(keep, keep)]
                        alpha[m] = alpha[m][keep]
                        b_ard[m] = b_ard[m][keep]

            # --- convergence check (sustained small relative change in z) ---
            if pruned:
                stable_count = 0
            elif prev_z is not None and prev_z.shape == z.shape:
                rel_change = np.linalg.norm(z - prev_z) / max(
                    np.linalg.norm(prev_z), 1e-300
                )
                stable_count = stable_count + 1 if rel_change < self.tol else 0
            prev_z = z.copy()
            if stable_count >= _PATIENCE:
                n_iter = iteration + 1
                break

        self.n_iter_ = n_iter
        self.n_components_ = k
        self._draw_posterior_samples(
            rng, z, cov_z, w, cov_w, a_ard, b_ard, a_tau, b_tau, d
        )
        self.weights_: list[np.ndarray] = list(w)
        self.view_relevance_: np.ndarray = np.array(alpha)
        return self

    # ------------------------------------------------------------------
    # Posterior sampling
    # ------------------------------------------------------------------

    def _draw_posterior_samples(
        self,
        rng: np.random.Generator,
        z: np.ndarray,
        cov_z: np.ndarray,
        w: list[np.ndarray],
        cov_w: list[np.ndarray],
        a_ard: np.ndarray,
        b_ard: list[np.ndarray],
        a_tau: np.ndarray,
        b_tau: np.ndarray,
        d: list[int],
    ) -> None:
        """Draw ``posterior_samples_`` from the fitted variational posterior.

        Each view's scalar noise is broadcast to every feature, matching the
        ``log_psi_{i}`` convention of the other probabilistic models.
        """
        s = self.n_posterior_samples
        m_views = len(w)
        k = z.shape[1]
        samples: dict[str, np.ndarray] = {}

        chol_z = np.linalg.cholesky(cov_z)
        z_noise = rng.standard_normal((s, *z.shape)) @ chol_z.T
        samples["z"] = z[np.newaxis, :, :] + z_noise

        tau_samples = np.stack(
            [rng.gamma(a_tau[m], 1.0 / b_tau[m], size=s) for m in range(m_views)],
            axis=1,
        )  # (S, M)
        alpha_samples = np.stack(
            [rng.gamma(a_ard[m], 1.0 / b_ard[m], size=(s, k)) for m in range(m_views)],
            axis=1,
        )  # (S, M, K)
        samples["alpha"] = alpha_samples

        for m in range(m_views):
            chol_w = np.linalg.cholesky(cov_w[m])
            w_noise = rng.standard_normal((s, d[m], k)) @ chol_w.T
            samples[f"W_{m}"] = w[m][np.newaxis, :, :] + w_noise
            psi_m = 1.0 / tau_samples[:, m]  # (S,)
            samples[f"log_psi_{m}"] = np.log(psi_m)[:, np.newaxis] * np.ones((1, d[m]))

        self.posterior_samples_ = samples
