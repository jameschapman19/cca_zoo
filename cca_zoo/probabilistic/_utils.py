"""Shared inference utilities for the probabilistic module."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike


def posterior_mean_latent(
    centered_views: list[np.ndarray],
    weights: list[np.ndarray],
    psi: list[np.ndarray],
) -> np.ndarray:
    r"""Posterior mean of the shared latent variable given fixed loadings.

    Closed-form posterior mean for a linear-Gaussian multiview factor model
    with point-estimate loadings ``weights`` and diagonal per-feature noise
    variances ``psi``:

    $$
    \Sigma_{z \mid x} = \left(I + \sum_i W_i^\top \Psi_i^{-1} W_i\right)^{-1}
    $$

    $$
    \mu_{z \mid x} = \Sigma_{z \mid x} \sum_i W_i^\top \Psi_i^{-1} x_i
    $$

    Args:
        centered_views: Per-view arrays, each ``(n_samples, n_features_i)``,
            already centred by the caller.
        weights: Per-view loading matrices, each ``(n_features_i, k)``.
        psi: Per-view noise-variance vectors, each ``(n_features_i,)``.

    Returns:
        Array of shape ``(n_samples, k)``: the posterior mean of z.
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


class PosteriorMeanTransformMixin:
    """Shared ``transform`` for models storing posterior samples of ``log_psi_i``.

    Requires the including class to set, after fitting: ``n_views_``,
    ``means_``, ``weights_`` (posterior mean loadings), and
    ``posterior_samples_`` (a dict with a ``log_psi_{i}`` entry per view,
    each of shape ``(num_samples, n_features_i)``).
    """

    # Declared for mypy: set by the including class's fit(), not here.
    n_views_: int
    means_: list[np.ndarray]
    weights_: list[np.ndarray]
    posterior_samples_: dict[str, Any]

    def transform(self, views: list[ArrayLike]) -> list[np.ndarray]:
        """Return the posterior mean of the shared latent variable z.

        Note:
            Unlike every other model in ``cca_zoo`` (one array per view),
            this returns a **single-element** list: this is a fully
            generative joint model with one shared latent variable rather
            than a per-view projection, so there is only one array to
            return.

        Args:
            views: List of arrays, each of shape (n_samples, n_features_i).

        Returns:
            List with exactly one numpy array of shape
            (n_samples, latent_dimensions) containing the posterior mean of
            the shared latent variable z for each observation.

        Raises:
            sklearn.exceptions.NotFittedError: If ``fit`` has not been called.
        """
        from sklearn.utils.validation import check_is_fitted

        from cca_zoo._utils._validation import validate_views

        check_is_fitted(self)
        validated = validate_views(views)
        centered = [v - m for v, m in zip(validated, self.means_)]

        psi = [
            np.exp(np.array(self.posterior_samples_[f"log_psi_{i}"])).mean(axis=0)
            for i in range(self.n_views_)
        ]
        return [posterior_mean_latent(centered, self.weights_, psi)]

    def _per_view_projections(self, views: list[ArrayLike]) -> list[np.ndarray]:
        """Project each view through its own posterior-mean loadings.

        ``transform`` returns a single shared latent array (there is one
        joint z, not one per view), which is what a caller wants for
        prediction but can't be compared *across* views the way every other
        model's per-view canonical variates can. Pairwise correlation needs
        a distinct representation per view, so this uses each view's own
        ``v_i @ W_i`` projection instead — analogous to every other model in
        the package, and to what ``transform`` would give without the
        cross-view precision weighting.
        """
        from cca_zoo._utils._validation import validate_views

        validated = validate_views(views)
        centered = [v - m for v, m in zip(validated, self.means_)]
        return [v @ w for v, w in zip(centered, self.weights_)]

    def pairwise_correlations(self, views: list[ArrayLike]) -> np.ndarray:
        """Compute the full pairwise correlation matrix per latent dimension.

        Uses each view's own posterior-mean projection (see
        :meth:`_per_view_projections`), not the shared-z ``transform``
        output, since the latter has no per-view distinction to correlate.

        Args:
            views: List of arrays, each of shape (n_samples, n_features_i).

        Returns:
            Array of shape ``(n_views, n_views, latent_dimensions)`` where
            entry ``[i, j, d]`` is the Pearson correlation between view i's
            and view j's own projection onto the d-th latent dimension.
        """
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self)
        per_view = self._per_view_projections(views)
        T = np.stack(per_view, axis=0)  # (n_views, n_samples, k)
        T = T - T.mean(axis=1, keepdims=True)
        norms = np.sqrt((T**2).sum(axis=1, keepdims=True))
        T_norm = T / np.where(norms > 1e-12, norms, 1.0)
        corrs: np.ndarray = np.einsum("isd,jsd->ijd", T_norm, T_norm)
        return corrs

    def average_pairwise_correlations(self, views: list[ArrayLike]) -> np.ndarray:
        """Return the mean off-diagonal pairwise correlation per dimension.

        Args:
            views: List of arrays, each of shape (n_samples, n_features_i).

        Returns:
            Array of shape ``(latent_dimensions,)`` with the average
            off-diagonal pairwise correlation for each canonical dimension.
        """
        corrs = self.pairwise_correlations(views)
        n_views = corrs.shape[0]
        off_diag_sum: np.ndarray = corrs.sum(axis=(0, 1)) - sum(
            corrs[i, i, :] for i in range(n_views)
        )
        n_pairs = n_views * (n_views - 1)
        result: np.ndarray = off_diag_sum / n_pairs
        return result

    def score(self, views: list[ArrayLike], y: None = None) -> np.ndarray:
        """Return average pairwise canonical correlations for each dimension.

        Args:
            views: List of arrays, each of shape (n_samples, n_features_i).
            y: Ignored.

        Returns:
            Array of shape ``(latent_dimensions,)`` with the average
            pairwise correlation for each canonical dimension.
        """
        return self.average_pairwise_correlations(views)

    def get_factor_loadings(self, views: list[ArrayLike]) -> list[np.ndarray]:
        """Compute canonical factor loadings for each view.

        Uses each view's own posterior-mean projection (see
        :meth:`_per_view_projections`), not the shared-z ``transform``
        output: the latter is a single array, so zipping it against every
        view (as :meth:`~cca_zoo._base.BaseModel.get_factor_loadings` does)
        would silently pair it with only the first view.

        Args:
            views: List of arrays, each of shape (n_samples, n_features_i).

        Returns:
            List of arrays, each of shape (n_features_i, latent_dimensions),
            where entry ``[j, d]`` is the correlation between feature j of
            view i and view i's own projection onto the d-th latent
            dimension.
        """
        from cca_zoo._utils._validation import validate_views

        validated = validate_views(views)
        per_view = self._per_view_projections(views)
        loadings = []
        for v, t in zip(validated, per_view):
            v_c = v - v.mean(axis=0)
            t_c = t - t.mean(axis=0)
            cov = v_c.T @ t_c / (v.shape[0] - 1)
            std_v = np.maximum(v_c.std(axis=0, ddof=1), 1e-12)
            std_t = np.maximum(t_c.std(axis=0, ddof=1), 1e-12)
            loadings.append(cov / np.outer(std_v, std_t))
        return loadings
