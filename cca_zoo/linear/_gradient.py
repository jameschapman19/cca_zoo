"""Gradient-descent CCA variants for high-dimensional and streaming data.

These methods replace the full covariance matrix eigendecomposition with
mini-batch Riemannian gradient descent on the Stiefel manifold, making them
suitable for large-scale problems where forming O(p²) covariance matrices
is infeasible.

All classes accept a ``batch_size`` parameter; set to ``None`` (default) for
full-batch updates, which matches the exact linear models but uses gradient
descent instead of a direct eigendecomposition.

Classes:
    PLS_EY: Eckart-Young PLS (stochastic power iteration).
    CCA_EY: Eckart-Young CCA (whitened EY).
    MCCA_EY: Multiview extension of CCA_EY (>=2 views).
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import ArrayLike

from cca_zoo._base import BaseModel
from cca_zoo._utils._linalg import svd_whiten
from cca_zoo._utils._validation import perview_parameter

logger = logging.getLogger(__name__)


def _stiefel_retract(W: np.ndarray) -> np.ndarray:
    """Project a matrix onto the Stiefel manifold via polar retraction.

    Computes the orthogonal polar factor of W: the nearest matrix to W with
    orthonormal columns.

    Args:
        W: Matrix of shape (p, k).

    Returns:
        Matrix of shape (p, k) with orthonormal columns.
    """
    U, _, Vt = np.linalg.svd(W, full_matrices=False)
    return np.asarray(U @ Vt)


def _riemannian_grad(euclidean_grad: np.ndarray, W: np.ndarray) -> np.ndarray:
    r"""Project a Euclidean gradient onto the tangent space of the Stiefel manifold.

    The Riemannian gradient removes components that point away from the manifold:

    .. math::

        G_R = G - W \,\mathrm{sym}(W^\top G)

    where :math:`\mathrm{sym}(A) = (A + A^\top) / 2`.

    Args:
        euclidean_grad: Euclidean gradient, shape (p, k).
        W: Current point on the manifold, shape (p, k).

    Returns:
        Riemannian gradient of the same shape.
    """
    sym = W.T @ euclidean_grad
    sym = (sym + sym.T) / 2.0
    return np.asarray(euclidean_grad - W @ sym)


# ---------------------------------------------------------------------------
# PLS_EY — Eckart-Young PLS
# ---------------------------------------------------------------------------


class PLS_EY(BaseModel):
    r"""Stochastic Eckart-Young PLS for large-scale data.

    Optimises the Eckart-Young (EY) objective for PLS by mini-batch
    Riemannian gradient descent on the Stiefel manifold:

    .. math::

        \min_{U, V \,:\, U^\top U = I,\, V^\top V = I}
            \left\| X_1 U - X_2 V \right\|_F^2

    which is equivalent to maximising :math:`\mathrm{tr}(U^\top X_1^\top X_2 V)`
    (the PLS objective).  At each step the Euclidean gradient is projected onto
    the tangent space of the Stiefel manifold, and the result is retracted back
    to the manifold via polar decomposition.

    Suitable for high-dimensional or streaming data where forming the full
    (p × p) cross-covariance matrix is too expensive.

    References:
        Gemp, I., McWilliams, B., Vernade, C., & Graepel, T. (2022).
        EigenGame Unloaded: When playing games is better than optimizing.
        *ICLR 2022*.

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means. Default True.
        learning_rate: Riemannian gradient step size. Default is 1e-2.
        max_iter: Number of gradient iterations. Default is 1000.
        batch_size: Mini-batch size.  ``None`` uses the full dataset.
        tol: Convergence tolerance on the objective change. Default is 1e-6.
        random_state: Seed for reproducibility.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((200, 500))
        >>> X2 = rng.standard_normal((200, 400))
        >>> model = PLS_EY(latent_dimensions=4, batch_size=64, random_state=0)
        >>> model.fit([X1, X2])
    """

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        learning_rate: float = 1e-2,
        max_iter: int = 1000,
        batch_size: int | None = None,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        super().__init__(latent_dimensions=latent_dimensions, center=center)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.tol = tol
        self.random_state = random_state

    def fit(self, views: list[ArrayLike], y: None = None) -> PLS_EY:
        """Fit PLS_EY by Riemannian gradient descent.

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
        n = self.n_samples_
        bs = n if self.batch_size is None else min(self.batch_size, n)

        # Initialise weights on Stiefel manifold
        W = [
            _stiefel_retract(rng.standard_normal((p, self.latent_dimensions)))
            for p in self.n_features_in_
        ]
        prev_obj = np.inf
        for iteration in range(self.max_iter):
            idx = rng.choice(n, bs, replace=False)
            batch = [v[idx] for v in views_]
            obj, W = self._step(batch, W)
            if abs(prev_obj - obj) < self.tol:
                logger.debug("PLS_EY converged at iteration %d", iteration)
                break
            prev_obj = obj
        self.weights_ = W
        return self

    def _step(
        self,
        batch: list[np.ndarray],
        W: list[np.ndarray],
    ) -> tuple[float, list[np.ndarray]]:
        """Perform one Riemannian gradient step.

        Args:
            batch: Mini-batch view arrays.
            W: Current weight matrices on the Stiefel manifold.

        Returns:
            Tuple of (objective value, updated weight matrices).
        """
        n = batch[0].shape[0]
        grads = self._euclidean_grads(batch, W, n)
        obj = self._objective(batch, W, n)
        W_new = []
        for i, (w, g) in enumerate(zip(W, grads)):
            rg = _riemannian_grad(g, w)
            w_next = _stiefel_retract(w - self.learning_rate * rg)
            W_new.append(w_next)
        return float(obj), W_new

    def _euclidean_grads(
        self,
        batch: list[np.ndarray],
        W: list[np.ndarray],
        n: int,
    ) -> list[np.ndarray]:
        r"""Compute Euclidean gradients of the EY loss.

        The EY loss is :math:`\sum_{i \neq j} \|X_i W_i - X_j W_j\|_F^2 / n`.
        Gradient w.r.t. :math:`W_i` is
        :math:`-2/n \sum_{j \neq i} X_i^\top X_j W_j`.

        Args:
            batch: Mini-batch arrays.
            W: Current weight matrices.
            n: Batch size.

        Returns:
            List of Euclidean gradient matrices.
        """
        scores = [b @ w for b, w in zip(batch, W)]  # (n, k) each
        grads = []
        for i, b in enumerate(batch):
            target = sum(scores[j] for j in range(len(batch)) if j != i)
            grads.append(-2.0 / n * b.T @ target)
        return grads

    def _objective(
        self,
        batch: list[np.ndarray],
        W: list[np.ndarray],
        n: int,
    ) -> float:
        """Compute the EY objective (sum of pairwise squared Frobenius norms).

        Args:
            batch: Mini-batch arrays.
            W: Current weight matrices.
            n: Batch size.

        Returns:
            Scalar objective value.
        """
        scores = [b @ w for b, w in zip(batch, W)]
        total = 0.0
        for i in range(len(batch)):
            for j in range(i + 1, len(batch)):
                diff = scores[i] - scores[j]
                total += float(np.sum(diff**2)) / n
        return total


# ---------------------------------------------------------------------------
# CCA_EY — Eckart-Young CCA (whitened)
# ---------------------------------------------------------------------------


class CCA_EY(PLS_EY):
    r"""Eckart-Young CCA for large-scale data.

    Equivalent to :class:`PLS_EY` but applies per-view PCA whitening before
    the gradient updates, so the resulting objective is the CCA correlation
    rather than covariance.  This makes the method applicable to views with
    very different scales.

    The whitening pre-processing is computed once at the start of ``fit``
    using the full data, then the gradient updates operate in the whitened
    space.

    References:
        Gemp, I., McWilliams, B., Vernade, C., & Graepel, T. (2022).
        EigenGame Unloaded: When playing games is better than optimizing.
        *ICLR 2022*.

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means. Default True.
        c: Ridge regularisation parameter(s) in ``[0, 1]``.  Default is 0
            (standard CCA whitening); increase for noisy high-dimensional data.
        learning_rate: Riemannian gradient step size. Default is 1e-2.
        max_iter: Number of gradient iterations. Default is 1000.
        batch_size: Mini-batch size.  ``None`` uses the full dataset.
        tol: Convergence tolerance. Default is 1e-6.
        random_state: Seed for reproducibility.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((200, 500))
        >>> X2 = rng.standard_normal((200, 400))
        >>> model = CCA_EY(latent_dimensions=4, c=0.1, batch_size=64, random_state=0)
        >>> model.fit([X1, X2])
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
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            latent_dimensions=latent_dimensions,
            center=center,
            learning_rate=learning_rate,
            max_iter=max_iter,
            batch_size=batch_size,
            tol=tol,
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
        # Whiten each view; store whitening matrices to back-project weights
        whitened = []
        self._whiten_mats: list[np.ndarray] = []
        for v, ci in zip(views_, c_):
            v_w, W_whiten = svd_whiten(v, ci)
            whitened.append(v_w)
            self._whiten_mats.append(W_whiten)

        rng = np.random.default_rng(self.random_state)
        n = self.n_samples_
        bs = n if self.batch_size is None else min(self.batch_size, n)
        latent_dims_clamped = min(
            self.latent_dimensions, *[w.shape[1] for w in whitened]
        )
        W_white = [
            _stiefel_retract(rng.standard_normal((w.shape[1], latent_dims_clamped)))
            for w in whitened
        ]
        prev_obj = np.inf
        for iteration in range(self.max_iter):
            idx = rng.choice(n, bs, replace=False)
            batch = [v[idx] for v in whitened]
            obj, W_white = self._step(batch, W_white)
            if abs(prev_obj - obj) < self.tol:
                logger.debug("CCA_EY converged at iteration %d", iteration)
                break
            prev_obj = obj
        # Back-project from whitened space to original space
        self.weights_ = [wm @ ww for wm, ww in zip(self._whiten_mats, W_white)]
        return self


# ---------------------------------------------------------------------------
# MCCA_EY — Multiview Eckart-Young CCA
# ---------------------------------------------------------------------------


class MCCA_EY(CCA_EY):
    r"""Eckart-Young multiview CCA for large-scale data (>=2 views).

    Extends :class:`CCA_EY` to handle more than two views by optimising
    the multiview EY loss:

    .. math::

        \min_{\{W_i\}} \sum_{i \neq j}
            \left\| \tilde{X}_i W_i - \tilde{X}_j W_j \right\|_F^2

    where :math:`\tilde{X}_i` are the whitened views, and all weight
    matrices are constrained to lie on the Stiefel manifold.

    References:
        Gemp, I., McWilliams, B., Vernade, C., & Graepel, T. (2022).
        EigenGame Unloaded: When playing games is better than optimizing.
        *ICLR 2022*.

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means. Default True.
        c: Ridge regularisation parameter(s) in ``[0, 1]``.  Default is 0.
        learning_rate: Riemannian gradient step size. Default is 1e-2.
        max_iter: Number of gradient iterations. Default is 1000.
        batch_size: Mini-batch size.  ``None`` uses the full dataset.
        tol: Convergence tolerance. Default is 1e-6.
        random_state: Seed for reproducibility.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((200, 500))
        >>> X2 = rng.standard_normal((200, 400))
        >>> X3 = rng.standard_normal((200, 300))
        >>> model = MCCA_EY(latent_dimensions=4, c=0.1, batch_size=64, random_state=0)
        >>> model.fit([X1, X2, X3])
    """

    def fit(self, views: list[ArrayLike], y: None = None) -> MCCA_EY:
        """Fit MCCA_EY for 2 or more views.

        Args:
            views: List of arrays, each (n_samples, n_features_i).
            y: Ignored.

        Returns:
            self: Fitted estimator.

        Raises:
            ValueError: If fewer than 2 views are provided.
            ValueError: If views have inconsistent numbers of samples.
        """
        super().fit(views, y)
        return self
