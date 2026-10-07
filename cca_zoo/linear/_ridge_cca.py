"""Ridge-regularised CCA."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike

from cca_zoo._base import BaseModel
from cca_zoo._utils._linalg import svd_whiten, truncated_svd
from cca_zoo._utils._param_constraints import RIDGE_PARAMETER
from cca_zoo._utils._validation import perview_parameter


class RidgeCCA(BaseModel):
    r"""Ridge-regularised CCA of two views (canonical ridge).

    $$
    \max_{w_1, w_2} w_1^\top X_1^\top X_2 w_2
    \quad \text{subject to} \quad
    w_i^\top \bigl((1 - c_i) X_i^\top X_i + c_i I\bigr) w_i = 1,
    $$

    solved by whitening each view and taking the SVD of the cross-covariance.
    ``shrinkage=0`` is :class:`CCA` and ``shrinkage=1`` is :class:`PLS`.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.
        shrinkage: Shrinkage of each view's covariance towards the identity,
            in ``[0, 1]``: 0 is CCA and 1 is PLS. Per-view. Default is 0.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).

    References:
        Vinod, H. D. (1976). Canonical ridge and econometrics of joint
        production. Journal of Econometrics, 4(2), 147-166.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.linear import RidgeCCA
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 10))
        >>> X2 = rng.standard_normal((50, 8))
        >>> model = RidgeCCA(n_components=2, shrinkage=[0.1, 0.5]).fit([X1, X2])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "shrinkage": RIDGE_PARAMETER,
    }
    _preserved_dtypes: ClassVar[list[type]] = [np.float64, np.float32]
    _supports_array_api: ClassVar[bool] = True

    def __init__(
        self,
        n_components: int = 1,
        *,
        center: bool = True,
        shrinkage: float | list[float] = 0.0,
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.shrinkage = shrinkage

    def _shrinkage_per_view(self) -> list[float]:
        return perview_parameter("shrinkage", self.shrinkage, 0.0, 2)

    def fit(
        self,
        views: list[ArrayLike],
        y: None = None,
        sample_weight: ArrayLike | None = None,
    ) -> RidgeCCA:
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
        views_: list[np.ndarray] = self._setup_fit(views, sample_weight)
        if self.n_views_ != 2:
            raise ValueError(
                f"{type(self).__name__} requires exactly 2 views, got "
                f"{self.n_views_}. Use MCCA for more than 2 views."
            )
        c_ = self._shrinkage_per_view()
        X1, X2 = views_
        # Whiten each view with its regularised covariance, then take the SVD
        # of the whitened views' cross-covariance.
        X1_w, W1 = svd_whiten(X1, c_[0])
        X2_w, W2 = svd_whiten(X2, c_[1])
        k = min(self.n_components, X1_w.shape[1], X2_w.shape[1])
        U, _, Vt = truncated_svd(X1_w.T @ X2_w / (X1.shape[0] - 1), k)
        self.weights_: list[Any] = [W1 @ U, W2 @ Vt.T]
        self._fit_maps_and_importances(views_)
        return self
