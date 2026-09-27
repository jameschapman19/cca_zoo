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


def posterior_mean_latent(
    centered_views: list[np.ndarray],
    weights: list[np.ndarray],
    psi: list[np.ndarray],
) -> np.ndarray:
    r"""Posterior mean of the shared latent given loadings and noise variances.

    $$
    \mu_{z \mid x} = \Bigl(I + \sum_i W_i^\top \Psi_i^{-1} W_i\Bigr)^{-1}
        \sum_i W_i^\top \Psi_i^{-1} x_i
    $$

    Args:
        centered_views: Centred arrays of shape (n_samples, n_features_i).
        weights: Loadings of shape (n_features_i, k), one per view.
        psi: Noise variances of shape (n_features_i,), one per view.

    Returns:
        The posterior mean, shape (n_samples, k).
    """
    k = weights[0].shape[1]
    n = centered_views[0].shape[0]
    precision = np.eye(k)
    information = np.zeros((n, k))
    for xi, w_i, psi_i in zip(centered_views, weights, psi):
        psi_inv = 1.0 / np.maximum(psi_i, 1e-8)
        precision = precision + w_i.T @ (w_i * psi_inv[:, np.newaxis])
        information = information + (xi * psi_inv) @ w_i
    sigma_z = np.linalg.inv(precision)
    return information @ sigma_z


def marginal_log_likelihood(
    centered_views: list[np.ndarray],
    weights: list[np.ndarray],
    psi: list[np.ndarray],
) -> float:
    r"""Mean per-sample log-likelihood with the latent integrated out.

    The concatenated views are $x \sim \mathcal{N}(0, \Psi + W W^\top)$,
    evaluated jointly so the cross-view covariance counts, via the Woodbury
    identity at cost linear in the total number of features.

    Args:
        centered_views: Centred arrays of shape (n_samples, n_features_i).
        weights: Loadings of shape (n_features_i, k), one per view.
        psi: Noise variances of shape (n_features_i,), one per view.

    Returns:
        The mean log-likelihood per sample.
    """
    w_full = np.concatenate(weights, axis=0)  # (P, k)
    psi_full = np.concatenate(psi, axis=0)  # (P,)
    x_full = np.concatenate(centered_views, axis=1)  # (n, P)
    n_samples, n_features = x_full.shape
    k = w_full.shape[1]

    psi_inv = 1.0 / np.maximum(psi_full, 1e-8)  # (P,)
    m = np.eye(k) + (w_full.T * psi_inv) @ w_full  # I + W^T Psi^-1 W, (k, k)
    m_inv = np.linalg.inv(m)

    # log det(Psi + W W^T) = log det(Psi) + log det(I + W^T Psi^-1 W)
    log_det_sigma = np.sum(np.log(np.maximum(psi_full, 1e-300)))
    log_det_sigma += np.linalg.slogdet(m)[1]

    x_scaled = x_full * psi_inv  # x^T Psi^-1, per sample: (n, P)
    quad_diag = np.einsum("np,np->n", x_full, x_scaled)  # x^T Psi^-1 x
    proj = x_scaled @ w_full  # (Psi^-1 x)^T W, per sample: (n, k)
    quad_correction = np.einsum("nk,kj,nj->n", proj, m_inv, proj)
    quad = quad_diag - quad_correction  # x^T Sigma^-1 x, via Woodbury

    log_lik_per_sample = -0.5 * (n_features * np.log(2 * np.pi) + log_det_sigma + quad)
    return float(np.mean(log_lik_per_sample))


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

    Subclasses set ``weights_`` and ``posterior_samples_``, with a
    ``log_psi_{i}`` entry of noise log-variances per view, in ``fit``.
    """

    weights_: list[np.ndarray]
    posterior_samples_: dict[str, Any]

    _components_bounded_by_features: ClassVar[bool] = False

    def _noise_variances(self) -> list[np.ndarray]:
        """Posterior-mean per-feature noise variance of each view."""
        return [
            np.exp(np.array(self.posterior_samples_[f"log_psi_{i}"])).mean(axis=0)
            for i in range(self.n_views_)
        ]

    def _encoder(self, view: int) -> np.ndarray:
        r"""Matrix mapping a centred view to its own posterior mean latent.

        $\Psi_i^{-1} W_i (I + W_i^\top \Psi_i^{-1} W_i)^{-1}$, of shape
        (n_features_i, k).
        """
        w = self.weights_[view]
        psi_inv = 1.0 / np.maximum(self._noise_variances()[view], 1e-8)
        scaled = w * psi_inv[:, np.newaxis]
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
        psi = self._noise_variances()
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
        return marginal_log_likelihood(centered, self.weights_, self._noise_variances())
