"""Full-batch L-BFGS-B fitting for Eckart-Young linear models."""

from __future__ import annotations

from abc import abstractmethod
from numbers import Integral, Real
from typing import Any, ClassVar

import numpy as np
from scipy.optimize import minimize
from sklearn.utils._param_validation import Interval

from cca_zoo._base import BaseModel
from cca_zoo._utils._ey import random_orthonormal_weights


class BaseFullBatchEYModel(BaseModel):
    """Base class for linear models fitted by L-BFGS-B on an EY-style loss.

    Subclasses implement :meth:`_objective` and :meth:`_derivative` and call
    :meth:`_fit_lbfgsb` from ``fit``.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.
        max_iter: Maximum L-BFGS-B iterations. Default is 1000.
        tol: L-BFGS-B ``ftol``, a relative per-step improvement; looser values
            can stop on a slow stretch of descent. Default is 1e-8.
        random_state: Seed for the initial weights. Default is None.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="neither")],
    }

    def __init__(
        self,
        n_components: int = 1,
        center: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-8,
        random_state: int | None = None,
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    @abstractmethod
    def _derivative(
        self,
        views: list[np.ndarray],
        representations: list[np.ndarray],
        weights: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Gradient of the loss in each view's weights."""

    @abstractmethod
    def _objective(
        self,
        views: list[np.ndarray],
        representations: list[np.ndarray],
        weights: list[np.ndarray],
    ) -> float:
        """The loss."""

    def _initial_weights(
        self, views: list[np.ndarray], rng: np.random.Generator
    ) -> list[np.ndarray]:
        """Random weights with orthonormal columns, one matrix per view."""
        return random_orthonormal_weights(views, self.n_components, rng)

    def _fit_lbfgsb(
        self, views: list[np.ndarray], rng: np.random.Generator
    ) -> list[np.ndarray]:
        """Weights minimising the loss by full-batch L-BFGS-B."""
        weights0 = self._initial_weights(views, rng)
        shapes = [w.shape for w in weights0]
        sizes = [w.size for w in weights0]

        def _unflatten(x: np.ndarray) -> list[np.ndarray]:
            arrays = []
            offset = 0
            for shape, size in zip(shapes, sizes):
                arrays.append(x[offset : offset + size].reshape(shape))
                offset += size
            return arrays

        def _fun(x: np.ndarray) -> tuple[float, np.ndarray]:
            weights = _unflatten(x)
            representations = [v @ w for v, w in zip(views, weights)]
            obj = self._objective(views, representations, weights)
            grads = self._derivative(views, representations, weights)
            grad = np.concatenate([g.ravel() for g in grads])
            return obj, grad

        x0 = np.concatenate([w.ravel() for w in weights0])
        result = minimize(
            _fun,
            x0,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )
        return _unflatten(result.x)
