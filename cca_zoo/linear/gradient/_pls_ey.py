"""PLS_EY — stochastic Eckart-Young PLS."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from cca_zoo._utils._ey import ey_cross_covariance
from cca_zoo.linear.gradient._base import BaseGradientModel


class PLS_EY(BaseGradientModel):
    r"""Stochastic Eckart-Young PLS for large-scale data.

    Optimises the unconstrained Eckart-Young (EY) objective for PLS by
    mini-batch momentum gradient descent, with no manifold projection step:
    the quadratic penalty on the weight Gram matrices below drives the
    weights towards (approximate) orthonormality at the optimum on its own.

    For :math:`M` views with weights :math:`W_i` and embeddings
    :math:`Z_i = X_i W_i`, define:

    .. math::

        A = \frac{1}{M} \sum_{i \neq j} \operatorname{Cov}(Z_i, Z_j), \qquad
        B = \frac{1}{M} \sum_i W_i^\top W_i

    and minimise :math:`\mathcal{L} = -2 \operatorname{tr}(A)
    + \operatorname{tr}(B B)`.

    Suitable for high-dimensional or streaming data where forming the full
    (p x p) cross-covariance matrix is too expensive.

    References:
        Chapman, J., Lawry Aguila, A., & Wells, L. (2022). A Generalised
        EigenGame with Extensions to Multiview Representation Learning.
        arXiv:2211.11323.

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means. Default True.
        learning_rate: Gradient step size. Default is 1e-2.
        max_iter: Number of gradient steps. Default is 1000.
        batch_size: Mini-batch size. ``None`` uses the full dataset.
        tol: Convergence tolerance on the objective change. Default is 1e-6.
        momentum: Momentum coefficient in ``[0, 1)``. Default is 0.9.
        random_state: Seed for reproducibility.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((200, 500))
        >>> X2 = rng.standard_normal((200, 400))
        >>> model = PLS_EY(latent_dimensions=4, batch_size=64, random_state=0)
        >>> model.fit([X1, X2])
    """

    def fit(self, views: list[ArrayLike], y: None = None) -> PLS_EY:
        """Fit PLS_EY by mini-batch momentum gradient descent.

        Args:
            views: List of arrays, each (n_samples, n_features_i).
            y: Ignored.

        Returns:
            self: Fitted estimator.

        Raises:
            ValueError: If fewer than 2 views are provided.
            ValueError: If views have inconsistent numbers of samples.
        """
        views_: list[np.ndarray] = self._setup_fit(views)
        rng = np.random.default_rng(self.random_state)
        self.weights_ = self._gradient_descent(views_, rng)
        return self

    def _weight_gram_mean(self, weights: list[np.ndarray]) -> np.ndarray:
        """Mean weight Gram matrix ``B = (1/M) sum_i W_i^T W_i``.

        Args:
            weights: Current weight matrices.

        Returns:
            Matrix of shape (k, k).
        """
        return sum(w.T @ w for w in weights) / len(weights)

    def _derivative(
        self,
        views: list[np.ndarray],
        representations: list[np.ndarray],
        weights: list[np.ndarray],
    ) -> list[np.ndarray]:
        r"""Analytic gradient :math:`\partial \mathcal{L} / \partial W_k`.

        .. math::

            \frac{\partial \mathcal{L}}{\partial W_k} =
                -\frac{4}{M(n-1)} X_k^\top \left(S - Z_k\right)
                + \frac{4}{M} W_k B

        where :math:`S = \sum_i Z_i` (all centred) and :math:`B` is the mean
        weight Gram matrix. Verified against finite-difference gradients for
        M = 2, 3, 4.

        Args:
            views: Mini-batch of view arrays.
            representations: Current embeddings.
            weights: Current weight matrices.

        Returns:
            List of gradient matrices, one per view.
        """
        m = len(views)
        n = views[0].shape[0]
        B = self._weight_gram_mean(weights)
        centred_reps = [z - z.mean(axis=0) for z in representations]
        total = sum(centred_reps)
        grads = []
        for k, (view, zk) in enumerate(zip(views, centred_reps)):
            view_c = view - view.mean(axis=0)
            other = total - zk
            reward_grad = -(4.0 / (m * (n - 1))) * (view_c.T @ other)
            penalty_grad = (4.0 / m) * (weights[k] @ B)
            grads.append(reward_grad + penalty_grad)
        return grads

    def _objective(
        self,
        views: list[np.ndarray],
        representations: list[np.ndarray],
        weights: list[np.ndarray],
    ) -> float:
        """Scalar PLS_EY loss, used only for the ``tol`` convergence check.

        Args:
            views: Mini-batch of view arrays (unused; kept for interface
                consistency with :meth:`_derivative`).
            representations: Current embeddings.
            weights: Current weight matrices.

        Returns:
            Scalar loss value.
        """
        del views
        C, V = ey_cross_covariance(representations)
        A = C - V  # exclude the i == j terms that ey_cross_covariance includes
        B = self._weight_gram_mean(weights)
        return float(-np.trace(2.0 * A) + np.trace(B @ B))
