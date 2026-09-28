"""Partial least squares."""

from __future__ import annotations

from numpy.typing import ArrayLike

from cca_zoo.linear._ridge_cca import RidgeCCA


class PLS(RidgeCCA):
    r"""Partial least squares of two views.

    $$
    \max_{w_1, w_2} w_1^\top X_1^\top X_2 w_2
    \quad \text{subject to} \quad \|w_i\|_2 = 1,
    $$

    the truncated SVD of the cross-covariance; :class:`RidgeCCA` with
    ``shrinkage=1``.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).

    References:
        Wold, H. (1975). Soft modelling by latent variables: the nonlinear
        iterative partial least squares (NIPALS) approach. Perspectives in
        Probability and Statistics, 117-142.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.linear import PLS
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 10))
        >>> X2 = rng.standard_normal((50, 8))
        >>> Z1, Z2 = PLS(n_components=2).fit_transform([X1, X2])
    """

    def __init__(
        self,
        n_components: int = 1,
        *,
        center: bool = True,
    ) -> None:
        super().__init__(
            n_components=n_components,
            center=center,
            shrinkage=1.0,
        )

    def fit(
        self,
        views: list[ArrayLike],
        y: None = None,
        sample_weight: ArrayLike | None = None,
    ) -> PLS:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.
            sample_weight: Weight of each sample; an integer weight is the
                same as repeating the sample. None weights samples equally.

        Returns:
            self.

        Raises:
            ValueError: If there are not exactly two views.
        """
        return super().fit(views, y, sample_weight)
