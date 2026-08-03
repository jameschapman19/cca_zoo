"""PartialCCA — CCA adjusted for confounding variables."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils.validation import check_is_fitted

from cca_zoo._utils._linalg import gevp
from cca_zoo._utils._validation import perview_parameter, validate_views
from cca_zoo.linear._mcca import MCCA


class PartialCCA(MCCA):
    r"""Partial Canonical Correlation Analysis.

    Extends CCA to account for confounding variables ``partials`` that may
    drive the correlation between views. Each view is first deconfounded by
    regressing out ``partials`` via least squares, and (ridge-regularised)
    CCA is then applied to the residuals, subject to the additional
    constraint that canonical weights are orthogonal to the confounds:

    $$
    \begin{aligned}
    w_{opt} = \underset{w}{\mathrm{argmax}}\ w_1^\top X_1^\top X_2 w_2 \\
    \text{subject to } w_i^\top X_i^\top X_i w_i = 1, \quad w_i^\top X_i^\top Z = 0
    \end{aligned}
    $$

    References:
        Rao, B. R. (1969). Partial canonical correlations. *Trabajos de
        Estadistica y de Investigacion Operativa*, 20(2-3), 211-219.

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means before fitting. Default True.
        c: Ridge regularisation parameter(s) applied to the deconfounded
            views. Either a scalar or a per-view list. Default is 0.
        eps: Small constant added to the eigenvalues of B to ensure
            positive definiteness. Default is 1e-6.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 10))
        >>> X2 = rng.standard_normal((50, 8))
        >>> Z = rng.standard_normal((50, 3))
        >>> model = PartialCCA(latent_dimensions=2).fit([X1, X2], partials=Z)
        >>> scores = model.transform([X1, X2], partials=Z)
    """

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        c: float | list[float] = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(
            latent_dimensions=latent_dimensions,
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
        """Fit the Partial CCA model.

        Args:
            views: List of arrays, each (n_samples, n_features_i).
            y: Ignored.
            partials: Confound array of shape (n_samples, n_confounds) to
                regress out of each view before fitting CCA. Required.

        Returns:
            self: Fitted estimator.

        Raises:
            ValueError: If ``partials`` is not provided.
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
        _, eigvecs = gevp(A, B, self.latent_dimensions)
        splits = np.cumsum([v.shape[1] for v in deconfounded])
        self.weights_: list[np.ndarray] = np.split(eigvecs, splits[:-1], axis=0)
        return self

    def transform(
        self,
        views: list[ArrayLike],
        partials: ArrayLike | None = None,
    ) -> list[np.ndarray]:
        """Project views into the latent space, optionally removing confounds.

        Args:
            views: List of arrays, each (n_samples, n_features_i).
            partials: Confound array matching the one used at fit time. If
                omitted, no deconfounding is applied (falls back to a plain
                linear projection), which keeps ``score``/``fit_transform``
                usable without threading ``partials`` through every call.

        Returns:
            List of arrays, each (n_samples, latent_dimensions).
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
        """Fit and then transform the training data.

        Args:
            views: List of arrays, each (n_samples, n_features_i).
            y: Ignored.
            partials: Confound array, passed through to both ``fit`` and
                ``transform``.

        Returns:
            List of arrays, each (n_samples, latent_dimensions).
        """
        return self.fit(views, y=y, partials=partials).transform(
            views, partials=partials
        )
