"""Shared inference utilities for the probabilistic module."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike
from sklearn.utils.validation import check_is_fitted

from cca_zoo._base import BaseModel


def _integer_seed(random_state: int | None) -> int:
    """An integer JAX seed from any ``random_state``."""
    return int(np.random.default_rng(random_state).integers(2**31 - 1))


def _noise_solve(psi: np.ndarray, a: np.ndarray) -> np.ndarray:
    """``Psi^{-1} a`` for noise given as variances (p,) or a covariance (p, p)."""
    if psi.ndim == 1:
        scaled: np.ndarray = a / np.maximum(psi, 1e-8).reshape(-1, *[1] * (a.ndim - 1))
        return scaled
    solved: np.ndarray = scipy.linalg.cho_solve(scipy.linalg.cho_factor(psi), a)
    return solved


def _noise_logdet(psi: np.ndarray) -> float:
    """``log det Psi`` for noise given as variances or a covariance."""
    if psi.ndim == 1:
        return float(np.sum(np.log(np.maximum(psi, 1e-300))))
    return float(np.linalg.slogdet(psi)[1])


def posterior_mean_latent(
    centered_views: list[np.ndarray],
    weights: list[np.ndarray],
    psi: list[np.ndarray],
) -> np.ndarray:
    r"""Posterior mean of the shared latent given loadings and noise.

    $$
    \mu_{z \mid x} = \Bigl(I + \sum_i W_i^\top \Psi_i^{-1} W_i\Bigr)^{-1}
        \sum_i W_i^\top \Psi_i^{-1} x_i
    $$

    Args:
        centered_views: Centred arrays of shape (n_samples, n_features_i).
        weights: Loadings of shape (n_features_i, k), one per view.
        psi: Noise of each view: variances of shape (n_features_i,) or a
            covariance of shape (n_features_i, n_features_i).

    Returns:
        The posterior mean, shape (n_samples, k).
    """
    k = weights[0].shape[1]
    precision = np.eye(k)
    information = np.zeros((centered_views[0].shape[0], k))
    for xi, w_i, psi_i in zip(centered_views, weights, psi):
        scaled = _noise_solve(psi_i, w_i)
        precision = precision + w_i.T @ scaled
        information = information + xi @ scaled
    mean: np.ndarray = np.linalg.solve(precision, information.T).T
    return mean


def marginal_log_likelihood(
    centered_views: list[np.ndarray],
    weights: list[np.ndarray],
    psi: list[np.ndarray],
) -> float:
    r"""Mean per-sample log-likelihood with the latent integrated out.

    The concatenated views are $x \sim \mathcal{N}(0, \Psi + W W^\top)$ with
    $\Psi$ block-diagonal, evaluated jointly so the cross-view covariance
    counts, via the Woodbury identity one view's noise at a time.

    Args:
        centered_views: Centred arrays of shape (n_samples, n_features_i).
        weights: Loadings of shape (n_features_i, k), one per view.
        psi: Noise of each view, as in :func:`posterior_mean_latent`.

    Returns:
        The mean log-likelihood per sample.
    """
    k = weights[0].shape[1]
    n_features = sum(w.shape[0] for w in weights)
    m = np.eye(k)  # I + W' Psi^-1 W
    projection = np.zeros((centered_views[0].shape[0], k))  # x' Psi^-1 W
    quad = np.zeros(centered_views[0].shape[0])  # x' Psi^-1 x
    log_det = 0.0
    for xi, w_i, psi_i in zip(centered_views, weights, psi):
        scaled = _noise_solve(psi_i, w_i)
        m = m + w_i.T @ scaled
        projection = projection + xi @ scaled
        quad = quad + np.einsum("np,pn->n", xi, _noise_solve(psi_i, xi.T))
        log_det += _noise_logdet(psi_i)
    log_det += float(np.linalg.slogdet(m)[1])
    quad = quad - np.einsum("nk,kn->n", projection, np.linalg.solve(m, projection.T))
    log_lik = -0.5 * (n_features * np.log(2 * np.pi) + log_det + quad)
    return float(np.mean(log_lik))


def maximum_likelihood_start(
    views: list[np.ndarray], n_components: int
) -> dict[str, np.ndarray]:
    r"""Bach and Jordan's maximum-likelihood PCCA, as starting values for inference.

    $W_i = \Sigma_{ii} U_i P^{1/2}$ and $\Psi_i = \Sigma_{ii} - W_i W_i^\top$,
    for canonical directions $U_i$ at unit variance and canonical
    correlations $P$: the maximum likelihood for two views, and MCCA's
    analogue for more. Keyed by the sample sites of :func:`pcca_model`.

    Args:
        views: Centred arrays of shape (n_samples, n_features_i).
        n_components: Latent dimension.

    Returns:
        Starting values of ``W_{i}``, ``noise_sd_{i}`` and ``noise_corr_{i}``.
    """
    from cca_zoo.linear import MCCA
    from cca_zoo.metrics import average_pairwise_correlations, pairwise_correlations

    arrays: list[ArrayLike] = list(views)
    mcca = MCCA(n_components, center=False).fit(arrays)
    scores = mcca.transform(arrays)
    rho = np.clip(average_pairwise_correlations(pairwise_correlations(scores)), 0, 0.99)
    start = {}
    for i, (view, weights, score) in enumerate(zip(views, mcca.weights_, scores)):
        sigma = view.T @ view / (len(view) - 1)
        w = sigma @ (weights / score.std(axis=0, ddof=1)) * np.sqrt(rho)
        psi = sigma - w @ w.T
        psi += 1e-6 * np.trace(psi) / len(psi) * np.eye(len(psi))
        sd = np.sqrt(np.diag(psi))
        start[f"W_{i}"] = w
        start[f"noise_sd_{i}"] = sd
        start[f"noise_corr_{i}"] = np.linalg.cholesky(psi / np.outer(sd, sd))
    return start


def pcca_model(
    views: list[np.ndarray], n_components: int, weight_scale: Any = 1.0
) -> None:
    r"""Numpyro model of probabilistic CCA, the latent integrated out.

    Bach and Jordan's model, $x_i = W_i z + \epsilon_i$ with
    $\epsilon_i \sim \mathcal{N}(0, \Psi_i)$ and $\Psi_i$ a full covariance,
    so that the latent explains only what the views share. Each $\Psi_i$ is
    ``noise_sd_{i}`` times an LKJ correlation ``noise_corr_{i}``. Integrating
    $z$ out, the stacked views are $\mathcal{N}(0, W W^\top + \Psi)$.

    Args:
        views: Centred arrays of shape (n_samples, n_features_i).
        n_components: Latent dimension.
        weight_scale: Prior standard deviation of the loadings, broadcast to
            each view's.
    """
    import jax.numpy as jnp
    import jax.scipy.linalg
    import numpyro
    import numpyro.distributions as dist

    loadings, noise = [], []
    for i, x in enumerate(views):
        p = x.shape[1]
        loadings.append(
            numpyro.sample(
                f"W_{i}",
                dist.Normal(
                    jnp.zeros((p, n_components)),
                    jnp.broadcast_to(weight_scale, (p, n_components)),
                ).to_event(2),
            )
        )
        sd = numpyro.sample(
            f"noise_sd_{i}", dist.LogNormal(jnp.zeros(p), jnp.ones(p)).to_event(1)
        )
        cholesky = sd[:, None] * (
            numpyro.sample(f"noise_corr_{i}", dist.LKJCholesky(p, 1.0))
            if p > 1
            else jnp.ones((1, 1))
        )
        noise.append(cholesky @ cholesky.T)
    w = jnp.concatenate(loadings, axis=0)
    covariance = w @ w.T + jax.scipy.linalg.block_diag(*noise)
    with numpyro.plate("n", views[0].shape[0]):
        numpyro.sample(
            "x",
            dist.MultivariateNormal(jnp.zeros(len(covariance)), covariance),
            obs=jnp.concatenate([jnp.asarray(x) for x in views], axis=1),
        )


def align_posterior_rotation(
    w_samples: np.ndarray, n_iter: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    r"""Align posterior draws of the loadings by generalized Procrustes.

    The likelihood is invariant to $W_i \to W_i R$ for orthogonal $R$, so
    unaligned draws partly cancel when averaged.

    Args:
        w_samples: Stacked loadings, shape (n_draws, P, k).
        n_iter: Alignment passes.

    Returns:
        ``(aligned, rotations)``: the rotated draws, and the rotation of each
        draw, shape (n_draws, k, k).
    """
    num_samples, _, k = w_samples.shape
    reference = w_samples.mean(axis=0)
    rotations = np.tile(np.eye(k), (num_samples, 1, 1))
    aligned = w_samples.copy()
    for _ in range(n_iter):
        for s in range(num_samples):
            r = scipy.linalg.orthogonal_procrustes(w_samples[s], reference)[0]
            rotations[s] = r
            aligned[s] = w_samples[s] @ r
        reference = aligned.mean(axis=0)
    return aligned, rotations


class BaseProbabilistic(BaseModel):
    """Posterior inference of the shared latent for the probabilistic models.

    Subclasses set ``weights_`` and ``posterior_samples_`` in ``fit``. The
    samples hold each view's noise standard deviations as ``noise_sd_{i}``
    and, for a full noise covariance, the Cholesky factor of its correlation
    as ``noise_corr_{i}``.
    """

    weights_: list[np.ndarray]
    posterior_samples_: dict[str, Any]

    _components_bounded_by_features: ClassVar[bool] = False

    def _noise(self) -> list[np.ndarray]:
        """Posterior-mean noise of each view: a covariance, or variances if diagonal."""
        noise = []
        for i in range(self.n_views_):
            sd = np.asarray(self.posterior_samples_[f"noise_sd_{i}"])
            correlation = self.posterior_samples_.get(f"noise_corr_{i}")
            if correlation is None:
                noise.append(np.mean(sd**2, axis=0))
            else:
                cholesky = sd[:, :, None] * np.asarray(correlation)
                noise.append(np.mean(cholesky @ cholesky.transpose(0, 2, 1), axis=0))
        return noise

    def _encoder(self, view: int) -> np.ndarray:
        r"""Matrix mapping a centred view to its own posterior mean latent.

        $\Psi_i^{-1} W_i (I + W_i^\top \Psi_i^{-1} W_i)^{-1}$, of shape
        (n_features_i, k).
        """
        w = self.weights_[view]
        scaled = _noise_solve(self._noise()[view], w)
        encoder: np.ndarray = scaled @ np.linalg.inv(np.eye(w.shape[1]) + w.T @ scaled)
        return encoder

    def _transform_view(self, view: int, centred: np.ndarray) -> np.ndarray:
        """Posterior mean of the latent given this view alone."""
        scores: np.ndarray = centred @ self._encoder(view)
        return scores

    def _feature_importances(self, views: list[np.ndarray]) -> list[np.ndarray]:
        """Variance share of each feature in its view's linear posterior mean."""
        return [
            train.var(axis=0) * np.sum(self._encoder(i) ** 2, axis=1)
            for i, train in enumerate(views)
        ]

    def _shared_latent(self, observed: dict[int, np.ndarray]) -> np.ndarray:
        """Posterior mean of the latent given the observed views alone."""
        psi = self._noise()
        views = list(observed)
        return posterior_mean_latent(
            [observed[i] for i in views],
            [self.weights_[i] for i in views],
            [psi[i] for i in views],
        )

    def posterior_mean(self, views: list[ArrayLike | None]) -> np.ndarray:
        """Posterior mean of the shared latent variable.

        Unlike :meth:`transform`, which infers the latent from each view
        separately, this combines every view's evidence. ``None`` marks an
        unobserved view.

        Args:
            views: One array of shape (n_samples, n_features_i) or None per view.

        Returns:
            The posterior mean, shape (n_samples, n_components).

        Raises:
            ValueError: If ``views`` has the wrong length or is all None.
        """
        check_is_fitted(self)
        if len(views) != self.n_views_:
            raise ValueError(
                f"Expected {self.n_views_} views (pass None for an "
                f"unobserved view), got {len(views)}."
            )
        observed = {
            i: self._check_view(i, v) - self.means_[i]
            for i, v in enumerate(views)
            if v is not None
        }
        if not observed:
            raise ValueError("At least one view must be observed.")
        return self._shared_latent(observed)

    def log_likelihood(self, views: list[ArrayLike]) -> float:
        """Mean per-sample marginal log-likelihood of the views; higher is better.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.

        Returns:
            The mean log-likelihood per sample.
        """
        validated = self._check_views(views)
        centered = [v - m for v, m in zip(validated, self.means_)]
        return marginal_log_likelihood(centered, self.weights_, self._noise())
