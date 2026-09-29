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
            X, Y, filter_params=True, **self._kernel_kwargs_[view]
        )
        return kernel

    def _feature_maps(
        self, views: list[np.ndarray]
    ) -> tuple[list[ArrayLike], list[np.ndarray]]:
        """Each training view's coordinates in its centred kernel feature space.

        Records ``views_fit_`` and each view's kernel and centerer.

        Returns:
            The feature maps, and the projections from a centred kernel row
            to feature-space coordinates that :meth:`_set_weights` needs.
        """
        m = self.n_views_
        self._kernel_kwargs_ = [
            {"metric": k, "gamma": g, "degree": d, "coef0": c, **(extra or {})}
            for k, g, d, c, extra in zip(
                perview_parameter("kernel", self.kernel, "linear", m),
                perview_parameter("gamma", self.gamma, None, m),
                perview_parameter("degree", self.degree, 3, m),
                perview_parameter("coef0", self.coef0, 1.0, m),
                perview_parameter("kernel_params", self.kernel_params, {}, m),
            )
        ]
        self.views_fit_: list[np.ndarray] = views
        self._centerers_: list[KernelCenterer] = []
        projections: list[np.ndarray] = []
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
            self._centerers_.append(centerer)
            projections.append(U / np.sqrt(lam))
            features.append(U * np.sqrt(lam))
        return features, projections

    def _set_weights(
        self, projections: list[np.ndarray], feature_weights: list[np.ndarray]
    ) -> None:
        """Dual weights from the linear model's weights on the feature maps."""
        self.weights_ = [P @ w for P, w in zip(projections, feature_weights)]

    def _transform_view(self, view: int, centred: np.ndarray) -> np.ndarray:
        kernel = self._centerers_[view].transform(
            self._kernel(view, centred, self.views_fit_[view])
        )
        scores: np.ndarray = kernel @ self.weights_[view]
        return scores
