"""Full-batch L-BFGS-B fitting for Eckart-Young linear models."""

from __future__ import annotations

from abc import abstractmethod
from numbers import Integral, Real
from typing import Any, ClassVar

import numpy as np
from scipy.optimize import minimize
from sklearn.utils._param_validation import Interval

from cca_zoo._base import BaseModel
from cca_zoo._utils._convergence import warn_if_not_converged
from cca_zoo._utils._param_constraints import RANDOM_STATE


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
        n_iter_: L-BFGS-B iterations run.
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="neither")],
        "random_state": RANDOM_STATE,
    }

    def __init__(
        self,
        n_components: int = 1,
        *,
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

    @abstractmethod
    def _initial_weights(
        self, views: list[np.ndarray], rng: np.random.Generator
    ) -> list[np.ndarray]:
        """Starting weights, one matrix per view."""

    def _weight_scale(self, view: np.ndarray) -> float:
        """Root mean eigenvalue of the view's constraint matrix, its covariance."""
        return float(np.sqrt(np.mean(view.var(axis=0, ddof=1))))

    def _fit_lbfgsb(
        self, views: list[np.ndarray], rng: np.random.Generator
    ) -> list[np.ndarray]:
        """Weights minimising the loss by full-batch L-BFGS-B.

        L-BFGS-B stops on an absolute gradient, and a view's gradient scales
        with its units, so the search runs over ``u_i = s_i w_i``, with
        ``s_i`` from :meth:`_weight_scale`: the same loss, but in units where
        every view's gradient is comparable.
        """
        scales = [self._weight_scale(v) or 1.0 for v in views]
        weights0 = self._initial_weights(views, rng)
        shapes = [w.shape for w in weights0]
        splits = np.cumsum([w.size for w in weights0])[:-1]

        def _weights(u: np.ndarray) -> list[np.ndarray]:
            return [
                part.reshape(shape) / s
                for part, shape, s in zip(np.split(u, splits), shapes, scales)
            ]

        def _fun(u: np.ndarray) -> tuple[float, np.ndarray]:
            weights = _weights(u)
            representations = [v @ w for v, w in zip(views, weights)]
            obj = self._objective(views, representations, weights)
            grads = self._derivative(views, representations, weights)
            grad = np.concatenate([(g / s).ravel() for g, s in zip(grads, scales)])
            return obj, grad

        u0 = np.concatenate([(w * s).ravel() for w, s in zip(weights0, scales)])
        result = minimize(
            _fun,
            u0,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )
        self.n_iter_: int = result.nit
        warn_if_not_converged(self, result.nit < self.max_iter)
        return _weights(result.x)
