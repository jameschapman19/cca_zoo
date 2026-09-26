"""Eckart-Young PLS."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from cca_zoo._utils._ey import random_orthonormal_weights
from cca_zoo.linear.gradient._cca_ey import CCAEY


class PLSEY(CCAEY):
    """Multiview PLS by minimising the Eckart-Young loss.

    :class:`~cca_zoo.linear.gradient.CCAEY` with ``c=1``, fitted by
    full-batch L-BFGS-B without forming a covariance matrix, so suited to
    wide data. See :class:`~cca_zoo.linear.gradient.StochasticCCAEY` for
    mini-batches.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.
        max_iter: Maximum L-BFGS-B iterations. Default is 1000.
        tol: L-BFGS-B ``ftol``. Default is 1e-8.
        random_state: Seed for the initial weights. Default is None.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).

    References:
        Chapman, J., Wells, L., & Lawry Aguila, A. (2024). Unconstrained
        Stochastic CCA: Unifying Multiview and Self-Supervised Learning.
        arXiv:2310.01012.

    Example:
        >>> import numpy as np
        >>> from cca_zoo.linear import PLSEY
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((200, 500))
        >>> X2 = rng.standard_normal((200, 400))
        >>> model = PLSEY(n_components=4, random_state=0).fit([X1, X2])
    """

    def __init__(
        self,
        n_components: int = 1,
        center: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-8,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            n_components=n_components,
            center=center,
            c=1.0,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )

    def fit(self, views: list[ArrayLike], y: None = None) -> PLSEY:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.

        Returns:
            self.
        """
        return super().fit(views, y)

    def _initial_weights(
        self, views: list[np.ndarray], rng: np.random.Generator
    ) -> list[np.ndarray]:
        """Random orthonormal weights, matching the penalty on the weight Gram."""
        return random_orthonormal_weights(views, self.n_components, rng)
