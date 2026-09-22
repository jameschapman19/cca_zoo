"""PLSEY — full-batch Eckart-Young PLS (c=1 special case of CCAEY)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils import deprecated

from cca_zoo._utils._ey import random_orthonormal_weights
from cca_zoo.linear.gradient._cca_ey import CCAEY


class PLSEY(CCAEY):
    r"""Eckart-Young PLS for 2 or more views.

    This is equivalent to :class:`~cca_zoo.linear.gradient.CCAEY` with
    ``c=1``: the reward excludes the $i = j$ terms that ``CCAEY``'s
    ($c=0$) reward includes, and the penalty is purely
    $\operatorname{tr}(BB)$ on the weight Gram matrix $B$, which
    drives the weights towards (approximate) orthonormality at the optimum
    on its own — no manifold projection step, and no upfront whitening.

    Suitable for high-dimensional data where forming the full ($p \times p$)
    cross-covariance matrix is too expensive. Fit by full-batch L-BFGS-B
    using the loss's exact analytic gradient; for mini-batch training on
    datasets too large for a full-batch gradient evaluation, see
    :class:`~cca_zoo.linear.gradient.StochasticCCAEY` (``c=1``).

    Initial weights have exactly orthonormal columns (unit-norm, mutually
    orthogonal) before any optimisation step, matching the shape of this
    loss's own penalty on $B$ — unlike :class:`~cca_zoo.linear.gradient.CCAEY`'s
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
        max_iter: Maximum number of L-BFGS-B iterations. Default is 1000.
        tol: Convergence tolerance, passed to L-BFGS-B as ``ftol``. Default
            is 1e-8 (see :class:`~cca_zoo.linear.gradient.CCAEY`'s docstring
            for why a loose ``ftol`` risks silent premature convergence).
        random_state: Seed for reproducibility.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((200, 500))
        >>> X2 = rng.standard_normal((200, 400))
        >>> model = PLSEY(latent_dimensions=4, random_state=0)
        >>> model = model.fit([X1, X2])
    """

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-8,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            latent_dimensions=latent_dimensions,
            center=center,
            c=1.0,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )

    def fit(self, views: list[ArrayLike], y: None = None) -> PLSEY:
        """Fit PLSEY by full-batch L-BFGS-B on the EY loss.

        Args:
            views: List of 2 or more arrays, each (n_samples, n_features_i).
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

        Overrides :class:`~cca_zoo.linear.gradient.CCAEY`'s data-informed
        default, since this loss's own penalty targets weight-space
        orthonormality rather than projection-space decorrelation.
        """
        return random_orthonormal_weights(views, self.latent_dimensions, rng)


@deprecated("Renamed to PLSEY for sklearn-style naming; use PLSEY instead.")
class PLS_EY(PLSEY):
    pass
