"""Partial CCA."""

from __future__ import annotations

from itertools import pairwise
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike

from cca_zoo._utils._linalg import block_diag, gevp
from cca_zoo._utils._validation import perview_parameter
from cca_zoo.linear._mcca import MCCA


class PartialCCA(MCCA):
    r"""CCA of the views after regressing out confounds.

    $$
    \max_{w} w_1^\top X_1^\top X_2 w_2
    \quad \text{subject to} \quad
    w_i^\top X_i^\top X_i w_i = 1, \quad w_i^\top X_i^\top Z = 0,
    $$

    for confounds $Z$ (``partials``). The constraints leave MCCA's
    eigenproblem on the views' partial covariance given the confounds,

    $$
    \Sigma_{XX|Z} = \Sigma_{XX} - \Sigma_{XZ} \Sigma_{ZZ}^{-1} \Sigma_{ZX},
    $$

    whose off-diagonal blocks form $A$ and whose diagonal blocks, with
    ``shrinkage``, form $B$, as :class:`MCCA` forms them from the covariance.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.
        shrinkage: Shrinkage of each view's covariance towards the identity,
            in ``[0, 1]``: 0 is CCA and 1 is PLS. Per-view. Default is 0.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).
        confound_betas_: Regression of each view on the centred confounds,
            shape (n_confounds, n_features_i).
        partials_mean_: Mean of the training confounds, shape (n_confounds,).

    References:
        Rao, B. R. (1969). Partial canonical correlations. Trabajos de
        Estadistica y de Investigacion Operativa, 20(2-3), 211-219.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.linear import PartialCCA
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 10))
        >>> X2 = rng.standard_normal((50, 8))
        >>> Z = rng.standard_normal((50, 3))
        >>> model = PartialCCA(n_components=2).fit([X1, X2], partials=Z)
        >>> Z1, Z2 = model.transform([X1, X2], partials=Z)
    """

    _supports_array_api: ClassVar[bool] = False

    def __init__(
        self,
        n_components: int = 1,
        *,
        center: bool = True,
        shrinkage: float | list[float] = 0.0,
    ) -> None:
        super().__init__(
            n_components=n_components,
            center=center,
            shrinkage=shrinkage,
            pca=False,
        )

    def fit(
        self,
        views: list[ArrayLike],
        y: None = None,
        partials: ArrayLike | None = None,
    ) -> PartialCCA:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.
            partials: Confounds, shape (n_samples, n_confounds).

        Returns:
            self.

        Raises:
            ValueError: If ``partials`` is not given.
        """
        if partials is None:
            raise ValueError("PartialCCA requires `partials` to be provided to fit().")
        views_ = self._setup_fit(views)
        self.partials_mean_: np.ndarray = np.asarray(partials, dtype=float).mean(axis=0)
        z = np.asarray(partials, dtype=float) - self.partials_mean_
        x = np.hstack(views_)
        x = x - x.mean(axis=0)
        betas = np.linalg.pinv(z.T @ z) @ (z.T @ x)
        # The views' partial covariance given the confounds.
        partial = (x.T @ x - (x.T @ z) @ betas) / (len(x) - 1)
        edges = np.cumsum([0, *self.n_features_per_view_])
        within = [partial[a:b, a:b] for a, b in pairwise(edges)]
        c_ = perview_parameter("shrinkage", self.shrinkage, 0.0, self.n_views_)
        A = (partial - block_diag(within)) / self.n_views_
        B = self._floored_blocks(
            [(1.0 - ci) * w + ci * np.eye(len(w)) for w, ci in zip(within, c_)]
        )
        _, eigvecs = gevp(A, B, self.n_components)
        self.weights_: list[np.ndarray] = np.split(eigvecs, edges[1:-1], axis=0)
        self.confound_betas_: list[np.ndarray] = np.split(betas, edges[1:-1], axis=1)
        self._fit_maps_and_importances(views_)
        return self

    def transform(
        self,
        views: list[ArrayLike],
        partials: ArrayLike | None = None,
    ) -> list[np.ndarray]:
        """Project views into the latent space, regressing out confounds if given.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            partials: Confounds, shape (n_samples, n_confounds); None projects
                the views unadjusted. Default is None.

        Returns:
            One array of shape (n_samples, n_components) per view.
        """
        if partials is None:
            return super().transform(views)
        validated = self._check_views(views)
        z = np.asarray(partials, dtype=float) - self.partials_mean_
        centred = [v - m for v, m in zip(validated, self.means_)]
        deconfounded = [v - z @ beta for v, beta in zip(centred, self.confound_betas_)]
        return [v @ w for v, w in zip(deconfounded, self.weights_)]

    def fit_transform(
        self,
        views: list[ArrayLike],
        y: None = None,
        partials: ArrayLike | None = None,
    ) -> list[np.ndarray]:
        """Fit, then transform the training views.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.
            partials: Confounds, shape (n_samples, n_confounds).

        Returns:
            One array of shape (n_samples, n_components) per view.
        """
        return self.fit(views, y=y, partials=partials).transform(
            views, partials=partials
        )
