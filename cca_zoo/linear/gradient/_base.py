"""Shared full-batch L-BFGS-B fit for EY-style linear models."""

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
    """Shared full-batch fit for Eckart-Young (EY) style linear models.

    Subclasses implement :meth:`_derivative` (analytic weight gradient) and
    :meth:`_objective` (scalar loss) and call :meth:`_fit_lbfgsb` from their
    own ``fit``. Both are already exact, analytic functions of the full
    dataset, so fitting is a single :func:`scipy.optimize.minimize`
    ``"L-BFGS-B"`` call over the flattened per-view weight matrices, rather
    than a hand-rolled gradient-descent loop. See :mod:`cca_zoo._utils._ey`
    for the shared EY-loss machinery used by the CCA-family subclasses.

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means. Default True.
        max_iter: Maximum number of L-BFGS-B iterations. Default is 1000.
        tol: Convergence tolerance, passed to L-BFGS-B as ``ftol``. Default
            is 1e-6.
        random_state: Seed for the initial weights.
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="neither")],
    }

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        super().__init__(latent_dimensions=latent_dimensions, center=center)
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
        """Analytic gradient of the loss w.r.t. each view's weight matrix.

        Args:
            views: Per-view arrays.
            representations: Current embeddings ``[v @ w for v, w in ...]``.
            weights: Current weight matrices.

        Returns:
            List of gradient matrices, one per view, matching the shape of
            the corresponding entry in ``weights``.
        """

    @abstractmethod
    def _objective(
        self,
        views: list[np.ndarray],
        representations: list[np.ndarray],
        weights: list[np.ndarray],
    ) -> float:
        """Scalar loss value."""

    def _initial_weights(
        self, views: list[np.ndarray], rng: np.random.Generator
    ) -> list[np.ndarray]:
        """Cheap, data-independent orthonormal initial weights, one per view.

        The default for this base class; :class:`~cca_zoo.linear.gradient.CCAEY`
        overrides this with a data-informed initialisation more appropriate
        to its own loss (see
        :func:`cca_zoo._utils._ey.cheap_orthonormal_projection_weights`).

        Args:
            views: Per-view arrays; only used for their feature counts.
            rng: Random generator.

        Returns:
            List of weight matrices, each (p_i, k) with orthonormal columns,
            where ``k = min(latent_dimensions, p_i)``.
        """
        return random_orthonormal_weights(views, self.latent_dimensions, rng)

    def _fit_lbfgsb(
        self, views: list[np.ndarray], rng: np.random.Generator
    ) -> list[np.ndarray]:
        """Fit weight matrices by full-batch L-BFGS-B on the exact EY loss.

        Args:
            views: List of arrays to fit on.
            rng: Random generator used for initialisation.

        Returns:
            List of fitted weight matrices, one per view.
        """
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
