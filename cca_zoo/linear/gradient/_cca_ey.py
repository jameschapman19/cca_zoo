"""CCA_EY — Eckart-Young CCA (whitened)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from cca_zoo._utils._ey import ey_grad_z, ey_loss
from cca_zoo._utils._linalg import svd_whiten
from cca_zoo._utils._validation import perview_parameter
from cca_zoo.linear.gradient._base import BaseGradientModel


class CCA_EY(BaseGradientModel):
    r"""Eckart-Young CCA for large-scale data.

    Applies per-view PCA whitening, then optimises the unconstrained
    Eckart-Young (EY) objective by mini-batch momentum gradient descent in
    the whitened space, with no manifold projection step. For embeddings
    :math:`Z_i` (from the whitened views), let :math:`C` be the mean
    pairwise cross-covariance (including :math:`i = j` terms) and :math:`V`
    the mean auto-covariance across views (see
    :func:`cca_zoo._utils._ey.ey_cross_covariance`); the loss minimised is

    .. math::

        \mathcal{L}_{EY} = -2 \operatorname{tr}(C) + \operatorname{tr}(V V)

    This objective has the canonical directions as a stationary point
    without requiring an explicit orthonormality constraint, unlike a plain
    squared-projection-distance loss.

    References:
        Chapman, J., Lawry Aguila, A., & Wells, L. (2022). A Generalised
        EigenGame with Extensions to Multiview Representation Learning.
        arXiv:2211.11323.

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means. Default True.
        c: Ridge regularisation parameter(s) in ``[0, 1]``.  Default is 0
            (standard CCA whitening); increase for noisy high-dimensional data.
        learning_rate: Gradient step size. Default is 1e-2.
        max_iter: Number of gradient steps. Default is 1000.
        batch_size: Mini-batch size.  ``None`` uses the full dataset.
        tol: Convergence tolerance. Default is 1e-6.
        momentum: Momentum coefficient in ``[0, 1)``. Default is 0.9.
        random_state: Seed for reproducibility.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((200, 500))
        >>> X2 = rng.standard_normal((200, 400))
        >>> model = CCA_EY(latent_dimensions=4, c=0.1, batch_size=64, random_state=0)
        >>> model = model.fit([X1, X2])
    """

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        c: float | list[float] = 0.0,
        learning_rate: float = 1e-2,
        max_iter: int = 1000,
        batch_size: int | None = None,
        tol: float = 1e-6,
        momentum: float = 0.9,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            latent_dimensions=latent_dimensions,
            center=center,
            learning_rate=learning_rate,
            max_iter=max_iter,
            batch_size=batch_size,
            tol=tol,
            momentum=momentum,
            random_state=random_state,
        )
        self.c = c

    def fit(self, views: list[ArrayLike], y: None = None) -> CCA_EY:
        """Fit CCA_EY with whitening pre-processing.

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
        c_ = perview_parameter("c", self.c, 0.0, self.n_views_)
        whitened = []
        self._whiten_mats: list[np.ndarray] = []
        for v, ci in zip(views_, c_):
            v_w, W_whiten = svd_whiten(v, ci)
            whitened.append(v_w)
            self._whiten_mats.append(W_whiten)

        rng = np.random.default_rng(self.random_state)
        W_white = self._gradient_descent(whitened, rng)
        # Back-project from whitened space to original feature space
        self.weights_ = [wm @ ww for wm, ww in zip(self._whiten_mats, W_white)]
        return self

    def _derivative(
        self,
        views: list[np.ndarray],
        representations: list[np.ndarray],
        weights: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Analytic gradient via the chain rule through the shared EY gradient.

        For a linear encoder ``Z_k = X_k @ W_k``, ``dL/dW_k = X_k^T @ dL/dZ_k``.

        Args:
            views: Mini-batch of whitened view arrays.
            representations: Current embeddings.
            weights: Current weight matrices (unused; the EY penalty depends
                only on the embeddings, not the weights directly).

        Returns:
            List of gradient matrices, one per view.
        """
        del weights
        grad_z = ey_grad_z(representations)
        return [view.T @ g for view, g in zip(views, grad_z)]

    def _objective(
        self,
        views: list[np.ndarray],
        representations: list[np.ndarray],
        weights: list[np.ndarray],
    ) -> float:
        """Scalar EY loss, used only for the ``tol`` convergence check."""
        del views, weights
        return ey_loss(representations)["objective"]
