"""Canonical correlation analysis."""

from __future__ import annotations

from numpy.typing import ArrayLike

from cca_zoo.linear._ridge_cca import RidgeCCA


class CCA(RidgeCCA):
    r"""Canonical correlation analysis of two views.

    $$
    \max_{w_1, w_2} w_1^\top X_1^\top X_2 w_2
    \quad \text{subject to} \quad w_i^\top X_i^\top X_i w_i = 1.
    $$

    :class:`RidgeCCA` with ``shrinkage=0``.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).

    References:
        Hotelling, H. (1936). Relations between two sets of variates.
        Biometrika, 28(3/4), 321-377.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.linear import CCA
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 10))
        >>> X2 = rng.standard_normal((50, 8))
        >>> Z1, Z2 = CCA(n_components=2).fit_transform([X1, X2])
    """

    def __init__(
        self,
        n_components: int = 1,
        center: bool = True,
    ) -> None:
        super().__init__(
            n_components=n_components,
            center=center,
            shrinkage=0.0,
        )

    def fit(self, views: list[ArrayLike], y: None = None) -> CCA:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.

        Returns:
            self.

        Raises:
            ValueError: If there are not exactly two views.
        """
        return super().fit(views, y)
