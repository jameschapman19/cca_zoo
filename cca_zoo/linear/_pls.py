"""PLS — Partial Least Squares (c=1 special case of rCCA)."""

from __future__ import annotations

from numpy.typing import ArrayLike

from cca_zoo.linear._rcca import rCCA


class PLS(rCCA):
    r"""Partial Least Squares (two-view).

    Finds the pair of unit-norm weight vectors that maximise the covariance
    between the projected views:

    .. math::

        \max_{\mathbf{w}_1, \mathbf{w}_2}
            \mathbf{w}_1^\top X_1^\top X_2 \mathbf{w}_2

        \text{subject to }
        \|\mathbf{w}_i\|_2 = 1

    This is equivalent to the truncated SVD of the sample cross-covariance
    matrix :math:`X_1^\top X_2 / (n - 1)`, and corresponds to :class:`rCCA`
    with ``c=1``.

    References:
        Wold, H. (1975). Soft modelling by latent variables: the nonlinear
        iterative partial least squares (NIPALS) approach. *Perspectives in
        Probability and Statistics*, 117–142.

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means before fitting. Default True.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 10))
        >>> X2 = rng.standard_normal((50, 8))
        >>> model = PLS(latent_dimensions=2).fit([X1, X2])
        >>> scores = model.transform([X1, X2])
    """

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
    ) -> None:
        super().__init__(
            latent_dimensions=latent_dimensions,
            center=center,
            c=1.0,
        )

    def fit(self, views: list[ArrayLike], y: None = None) -> PLS:
        """Fit the PLS model.

        Args:
            views: List of exactly two arrays, each (n_samples, n_features_i).
            y: Ignored.

        Returns:
            self: Fitted estimator.

        Raises:
            ValueError: If the number of views is not exactly 2.
            ValueError: If views have inconsistent numbers of samples.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> X1 = rng.standard_normal((50, 10))
            >>> X2 = rng.standard_normal((50, 8))
            >>> model = PLS(latent_dimensions=2).fit([X1, X2])
        """
        return super().fit(views, y)
