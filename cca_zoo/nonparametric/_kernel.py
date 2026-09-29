"""Shared kernel machinery of the kernel CCA models."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import pairwise_kernels
from sklearn.preprocessing import KernelCenterer

from cca_zoo._base import BaseModel
from cca_zoo._utils._param_constraints import KERNEL_PARAMETERS
from cca_zoo._utils._validation import perview_parameter


class _BaseKernelModel(BaseModel):
    """A linear model fitted to each view's kernel feature map.

    Each view's kernel is centred in feature space with sklearn's
    :class:`~sklearn.preprocessing.KernelCenterer` and eigendecomposed,
    ``K_c = U diag(lam) U'``, giving training coordinates ``U sqrt(lam)``
    whose inner products are ``K_c``. The linear model is fitted to these
    coordinates, and a new row's are its centred kernel against the training
    rows times ``U diag(lam)^{-1/2}``. The model is therefore exactly its
    linear counterpart in the kernel's (centred) feature space, with
    ``shrinkage`` meaning the same as for the linear models.
    """

    _components_bounded_by_features: ClassVar[bool] = False
    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        **KERNEL_PARAMETERS,
    }

    kernel: str | list[str]
    gamma: float | list[float | None] | None
    degree: float | list[float]
    coef0: float | list[float]
    kernel_params: dict[str, object] | list[dict[str, object]] | None
    weights_: list[np.ndarray]

    def _kernel(self, view: int, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        kernel: np.ndarray = pairwise_kernels(
            X,
            Y,
            metric=self._kernels[view],
            gamma=self._gammas[view],
            degree=self._degrees[view],
            coef0=self._coef0s[view],
            filter_params=True,
            **self._kernel_params[view],
        )
        return kernel

    def _feature_maps(self, views: list[np.ndarray]) -> list[ArrayLike]:
        """Each training view's coordinates in its centred kernel feature space.

        Records ``views_fit_``, the kernel centerers and ``_projections``,
        the maps from a centred kernel row to feature-space coordinates.
        """
        m = self.n_views_
        self._kernels = perview_parameter("kernel", self.kernel, "linear", m)
        self._gammas = perview_parameter("gamma", self.gamma, None, m)
        self._degrees = perview_parameter("degree", self.degree, 3, m)
        self._coef0s = perview_parameter("coef0", self.coef0, 1.0, m)
        self._kernel_params = [
            kp or {}
            for kp in perview_parameter("kernel_params", self.kernel_params, {}, m)
        ]
        self.views_fit_: list[np.ndarray] = views
        self._centerers: list[KernelCenterer] = []
        self._projections: list[np.ndarray] = []
        features: list[ArrayLike] = []
        for i, v in enumerate(views):
            kernel = self._kernel(i, v, v)
            centerer = KernelCenterer().fit(kernel)
            lam, U = np.linalg.eigh(centerer.transform(kernel))
            # Centring leaves a null direction, the constant vector, and
            # low-rank kernels leave more; keep the numerically positive
            # spectrum, as numpy.linalg.matrix_rank does.
            keep = lam > lam.max() * len(lam) * np.finfo(lam.dtype).eps
            lam, U = lam[keep], U[:, keep]
            self._centerers.append(centerer)
            self._projections.append(U / np.sqrt(lam))
            features.append(U * np.sqrt(lam))
        return features

    def _set_weights(self, feature_weights: list[np.ndarray]) -> None:
        """Dual weights from the linear model's weights on the feature maps."""
        self.weights_ = [P @ w for P, w in zip(self._projections, feature_weights)]

    def _transform_view(self, view: int, centred: np.ndarray) -> np.ndarray:
        kernel = self._centerers[view].transform(
            self._kernel(view, centred, self.views_fit_[view])
        )
        scores: np.ndarray = kernel @ self.weights_[view]
        return scores
