"""PLS_EY — stochastic Eckart-Young PLS (c=1 special case of CCA_EY)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from cca_zoo._utils._ey import random_orthonormal_weights
from cca_zoo.linear.gradient._cca_ey import CCA_EY


class PLS_EY(CCA_EY):
    r"""Stochastic Eckart-Young PLS for large-scale data.

    This is equivalent to :class:`~cca_zoo.linear.gradient.CCA_EY` with
    ``c=1``: the reward excludes the $i = j$ terms that ``CCA_EY``'s
    ($c=0$) reward includes, and the penalty is purely
    $\operatorname{tr}(BB)$ on the weight Gram matrix $B$, which
    drives the weights towards (approximate) orthonormality at the optimum
    on its own — no manifold projection step, and no upfront whitening.

    Suitable for high-dimensional or streaming data where forming the full
    (p x p) cross-covariance matrix is too expensive.

    Initial weights have exactly orthonormal columns (unit-norm, mutually
    orthogonal) before any gradient step, matching the shape of this loss's
    own penalty on $B$ — unlike :class:`~cca_zoo.linear.gradient.CCA_EY`'s
    own data-informed default, which instead orthonormalises the initial
    *projections* (see :func:`cca_zoo._utils._ey.random_orthonormal_weights`
    vs. :func:`cca_zoo._utils._ey.cheap_orthonormal_projection_weights`).

    References:
        Chapman, J., Wells, L., & Lawry Aguila, A. (2024). Unconstrained
        Stochastic CCA: Unifying Multiview and Self-Supervised Learning.
        arXiv:2310.01012.

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
        >>> model = model.fit([X1, X2])
    """

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
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
            c=1.0,
            learning_rate=learning_rate,
            max_iter=max_iter,
            batch_size=batch_size,
            tol=tol,
            momentum=momentum,
            random_state=random_state,
        )

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
        return super().fit(views, y)

    def _initial_weights(
        self, views: list[np.ndarray], rng: np.random.Generator
    ) -> list[np.ndarray]:
        """Plain orthonormal-weight initial weights (see class docstring).

        Overrides :class:`~cca_zoo.linear.gradient.CCA_EY`'s data-informed
        default, since this loss's own penalty targets weight-space
        orthonormality rather than projection-space decorrelation.
        """
        return random_orthonormal_weights(views, self.latent_dimensions, rng)
