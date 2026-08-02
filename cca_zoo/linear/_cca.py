"""CCA — standard Canonical Correlation Analysis (c=0 special case of rCCA)."""

from __future__ import annotations

from numpy.typing import ArrayLike

from cca_zoo.linear._rcca import rCCA


class CCA(rCCA):
    r"""Canonical Correlation Analysis.

    Finds the pair of linear projections that maximise the Pearson correlation
    between two views subject to unit within-view variance constraints:

    $$
    \begin{aligned}
    \max_{\mathbf{w}_1, \mathbf{w}_2} \mathbf{w}_1^\top X_1^\top X_2 \mathbf{w}_2 \\
    \text{subject to } \mathbf{w}_i^\top X_i^\top X_i \mathbf{w}_i = 1
    \end{aligned}
    $$

    This is a special case of :class:`rCCA` with ``c=0``.  The solution uses
    PCA whitening followed by an SVD of the cross-covariance matrix, which is
    numerically stable even for high-dimensional views.

    References:
        Hotelling, H. (1936). Relations between two sets of variates.
        *Biometrika*, 28(3/4), 321–377.

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means before fitting. Default True.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 10))
        >>> X2 = rng.standard_normal((50, 8))
        >>> model = CCA(latent_dimensions=2).fit([X1, X2])
        >>> corrs = model.score([X1, X2])
    """

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
    ) -> None:
        super().__init__(
            latent_dimensions=latent_dimensions,
            center=center,
            c=0.0,
        )

    def fit(self, views: list[ArrayLike], y: None = None) -> CCA:
        """Fit the CCA model.

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
            >>> model = CCA(latent_dimensions=2).fit([X1, X2])
        """
        return super().fit(views, y)
