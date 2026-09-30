"""Group Factor Analysis, ported from the R package CCAGFA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike

from cca_zoo._utils._convergence import warn_if_not_converged
from cca_zoo._utils._param_constraints import POSITIVE_EPS, POSITIVE_INT, RANDOM_STATE
from cca_zoo.probabilistic._utils import BaseProbabilistic

# CCAGFA::getDefaultOpts() priors: near-improper/flat, matching the R
# package's defaults exactly (prior.alpha_0/beta_0/alpha_0t/beta_0t <- 1e-14).
_ARD_ALPHA_0 = 1e-14
_ARD_BETA_0 = 1e-14
_TAU_ALPHA_0 = 1e-14
_TAU_BETA_0 = 1e-14
_INIT_TAU = 1e3
_DROP_TOL = 1e-7
_PATIENCE = 1000


@dataclass
class _VariationalPosterior:
    """Klami et al.'s factorised posterior q(Z) q(W) q(alpha) q(tau).

    Each ``update_*`` method is one closed-form coordinate-ascent step,
    following ``CCAGFA::GFA()``.
    """

    z: np.ndarray
    cov_z: np.ndarray
    w: list[np.ndarray]
    cov_w: list[np.ndarray]
    alpha: list[np.ndarray]
    tau: np.ndarray
    a_ard: np.ndarray
    b_ard: list[np.ndarray]
    a_tau: np.ndarray
    b_tau: np.ndarray
    sum_squares: np.ndarray

    @classmethod
    def initialise(
        cls, views: list[np.ndarray], k: int, rng: np.random.Generator
    ) -> _VariationalPosterior:
        """CCAGFA's initialisation, with its ``getDefaultOpts()`` priors."""
        n = views[0].shape[0]
        d = np.array([v.shape[1] for v in views])
        tau = np.full(len(views), _INIT_TAU)
        total_variance = np.array([np.var(v, axis=0, ddof=1).sum() for v in views])
        return cls(
            z=rng.standard_normal((n, k)),
            cov_z=np.eye(k),
            w=[np.zeros((p, k)) for p in d],
            cov_w=[np.eye(k) for _ in d],
            alpha=[
                np.full(k, k * p / max(var - 1.0 / t, 1e-8))
                for p, var, t in zip(d, total_variance, tau)
            ],
            tau=tau,
            a_ard=_ARD_ALPHA_0 + d / 2.0,
            b_ard=[np.full(k, _ARD_BETA_0) for _ in d],
            a_tau=_TAU_ALPHA_0 + n * d / 2.0,
            b_tau=np.full(len(views), _TAU_BETA_0),
            sum_squares=np.array([np.sum(v**2) for v in views]),
        )

    @property
    def zz(self) -> np.ndarray:
        """E[Z'Z]."""
        zz: np.ndarray = self.z.T @ self.z + len(self.z) * self.cov_z
        return zz

    def ww(self, m: int) -> np.ndarray:
        """E[W_m' W_m]."""
        ww: np.ndarray = self.w[m].T @ self.w[m] + len(self.w[m]) * self.cov_w[m]
        return ww

    def update_loadings(self, views: list[np.ndarray]) -> None:
        """q(W_m) for each view, given q(Z)."""
        k = self.z.shape[1]
        zz = self.zz
        for m, x in enumerate(views):
            scale = 1.0 / np.sqrt(self.alpha[m])
            inner = np.outer(scale, scale) * zz + np.eye(k) / self.tau[m]
            cho = np.linalg.cholesky(inner)
            inv_inner = np.linalg.solve(cho.T, np.linalg.solve(cho, np.eye(k)))
            self.cov_w[m] = np.outer(scale, scale) * inv_inner / self.tau[m]
            self.w[m] = x.T @ self.z @ self.cov_w[m] * self.tau[m]

    def update_factors(self, views: list[np.ndarray]) -> None:
        """q(Z), given every view's q(W)."""
        k = self.z.shape[1]
        precision = np.eye(k) + sum(t * self.ww(m) for m, t in enumerate(self.tau))
        cho = np.linalg.cholesky(precision)
        self.cov_z = np.linalg.solve(cho.T, np.linalg.solve(cho, np.eye(k)))
        self.z = sum(x @ w * t for x, w, t in zip(views, self.w, self.tau)) @ self.cov_z

    def update_ard(self) -> None:
        """q(alpha_m), the ARD precision of each view's loadings."""
        for m in range(len(self.w)):
            self.b_ard[m] = _ARD_BETA_0 + np.diag(self.ww(m)) / 2.0
            self.alpha[m] = self.a_ard[m] / self.b_ard[m]

    def update_noise(self, views: list[np.ndarray]) -> None:
        """q(tau_m), the noise precision of each view."""
        zz = self.zz
        for m, x in enumerate(views):
            residual = (
                self.sum_squares[m]
                + np.sum(self.ww(m) * zz)
                - 2.0 * np.sum(self.z * (x @ self.w[m]))
            )
            self.b_tau[m] = _TAU_BETA_0 + residual / 2.0
            self.tau[m] = self.a_tau[m] / self.b_tau[m]

    def prune(self) -> bool:
        """Drop dimensions whose factors have vanished, as CCAGFA's ``dropK``."""
        keep = np.where(np.mean(self.z**2, axis=0) > _DROP_TOL)[0]
        if not 0 < len(keep) < self.z.shape[1]:
            return False
        square = np.ix_(keep, keep)
        self.z = self.z[:, keep]
        self.cov_z = self.cov_z[square]
        for m in range(len(self.w)):
            self.w[m] = self.w[m][:, keep]
            self.cov_w[m] = self.cov_w[m][square]
            self.alpha[m] = self.alpha[m][keep]
            self.b_ard[m] = self.b_ard[m][keep]
        return True

    def sample(self, rng: np.random.Generator, s: int) -> dict[str, np.ndarray]:
        """Draws from q, keyed as the other probabilistic models' samples.

        Each view's scalar noise is broadcast to every feature as
        ``noise_sd_{i}``, diagonal noise in their convention.
        """
        k = self.z.shape[1]
        noise = (
            rng.standard_normal((s, *self.z.shape)) @ np.linalg.cholesky(self.cov_z).T
        )
        samples = {
            "z": self.z[np.newaxis] + noise,
            "alpha": np.stack(
                [
                    rng.gamma(a, 1.0 / b, size=(s, k))
                    for a, b in zip(self.a_ard, self.b_ard)
                ],
                axis=1,
            ),
        }
        for m, (w, cov) in enumerate(zip(self.w, self.cov_w)):
            noise = rng.standard_normal((s, *w.shape)) @ np.linalg.cholesky(cov).T
            samples[f"W_{m}"] = w[np.newaxis] + noise
            tau = rng.gamma(self.a_tau[m], 1.0 / self.b_tau[m], size=s)
            samples[f"noise_sd_{m}"] = np.repeat(tau[:, None] ** -0.5, len(w), axis=1)
        return samples


class GFA(BaseProbabilistic):
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
        ard_precision_: Posterior mean ARD precisions, shape
            (n_views, n_components_); large means shrunk away.
        n_components_: Number of dimensions kept.
        posterior_samples_: Posterior draws keyed ``W_{i}``, ``noise_sd_{i}``
            and ``alpha``.
        n_iter_: Number of iterations run.

    References:
        Klami, A., Virtanen, S., & Kaski, S. (2013). Bayesian Canonical
        Correlation Analysis. Journal of Machine Learning Research, 14,
        965-1003.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.probabilistic import GFA
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 4))
        >>> X2 = rng.standard_normal((50, 3))
        >>> model = GFA(n_components=2, max_iter=50).fit([X1, X2])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseProbabilistic._parameter_constraints,
        "max_iter": POSITIVE_INT,
        "tol": POSITIVE_EPS,
        "drop_k": ["boolean"],
        "n_posterior_samples": POSITIVE_INT,
        "random_state": RANDOM_STATE,
    }

    def __init__(
        self,
        n_components: int = 1,
        *,
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

    def fit(self, views: list[ArrayLike], y: None = None) -> GFA:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.

        Returns:
            self.
        """
        views_ = self._setup_fit(views)
        rng = np.random.default_rng(self.random_state)
        q = _VariationalPosterior.initialise(views_, self.n_components, rng)
        # Convergence needs `_PATIENCE` consecutive small changes in Z and no
        # pruning: the change can dip below tol for hundreds of iterations
        # while one dimension's ARD decay waits on another's.
        previous: np.ndarray | None = None
        stable, converged, self.n_iter_ = 0, False, self.max_iter
        for iteration in range(self.max_iter):
            q.update_loadings(views_)
            q.update_factors(views_)
            q.update_ard()
            q.update_noise(views_)
            if self.drop_k and q.prune():
                stable = 0
            elif previous is not None and previous.shape == q.z.shape:
                change = np.linalg.norm(q.z - previous) / np.linalg.norm(previous)
                stable = stable + 1 if change < self.tol else 0
            previous = q.z.copy()
            if stable >= _PATIENCE:
                converged, self.n_iter_ = True, iteration + 1
                break
        warn_if_not_converged(self, converged)

        self.weights_: list[np.ndarray] = q.w
        self.ard_precision_: np.ndarray = np.array(q.alpha)
        self.posterior_samples_ = q.sample(rng, self.n_posterior_samples)
        self._fit_maps_and_importances(views_)
        return self
