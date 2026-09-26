"""Partial CCA."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils.validation import check_is_fitted

from cca_zoo._utils._linalg import gevp
from cca_zoo._utils._validation import perview_parameter, validate_views
from cca_zoo.linear._mcca import MCCA


class PartialCCA(MCCA):
    r"""CCA of the views after regressing out confounds.

    $$
    \max_{w} w_1^\top X_1^\top X_2 w_2
    \quad \text{subject to} \quad
    w_i^\top X_i^\top X_i w_i = 1, \quad w_i^\top X_i^\top Z = 0,
    $$

    for confounds $Z$ (``partials``), solved as ridge CCA of the
    residuals of each view regressed on $Z$.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.
        c: Ridge blend in ``[0, 1]``. Per-view. Default is 0.
        eps: Floor added to the eigenvalues of ``B``. Default is 1e-6.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).
        confound_betas_: Regression of each view on the confounds, shape
            (n_confounds, n_features_i).

    References:
        Rao, B. R. (1969). Partial canonical correlations. Trabajos de
        Estadistica y de Investigacion Operativa, 20(2-3), 211-219.

    Example:
        >>> import numpy as np
        >>> from cca_zoo.linear import PartialCCA
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 10))
        >>> X2 = rng.standard_normal((50, 8))
        >>> Z = rng.standard_normal((50, 3))
        >>> model = PartialCCA(n_components=2).fit([X1, X2], partials=Z)
        >>> Z1, Z2 = model.transform([X1, X2], partials=Z)
    """

    def __init__(
        self,
        n_components: int = 1,
        center: bool = True,
        c: float | list[float] = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(
            n_components=n_components,
            center=center,
            c=c,
            pca=False,
            eps=eps,
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
        partials_arr = np.asarray(partials, dtype=float)
        self.confound_betas_: list[np.ndarray] = [
            np.linalg.pinv(partials_arr) @ v for v in views_
        ]
        deconfounded = [
            v - partials_arr @ beta for v, beta in zip(views_, self.confound_betas_)
        ]
        c_ = perview_parameter("c", self.c, 0.0, self.n_views_)
        A = self._build_A(deconfounded)
        B = self._build_B(deconfounded, c_)
        _, eigvecs = gevp(A, B, self.n_components)
        splits = np.cumsum([v.shape[1] for v in deconfounded])
        self.weights_: list[np.ndarray] = np.split(eigvecs, splits[:-1], axis=0)
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
        check_is_fitted(self)
        if partials is None:
            return super().transform(views)
        validated = validate_views(views)
        partials_arr = np.asarray(partials, dtype=float)
        centred = [v - m for v, m in zip(validated, self.means_)]
        deconfounded = [
            v - partials_arr @ beta for v, beta in zip(centred, self.confound_betas_)
        ]
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
