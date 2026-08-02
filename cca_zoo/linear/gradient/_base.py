"""Shared mini-batch momentum-SGD training loop for EY-style gradient models."""

from __future__ import annotations

from abc import abstractmethod

import numpy as np

from cca_zoo._base import BaseModel


class BaseGradientModel(BaseModel):
    """Shared mini-batch momentum-SGD loop for Eckart-Young (EY) style models.

    Subclasses implement :meth:`_derivative` (analytic weight gradient) and
    :meth:`_objective` (scalar loss, used only for the ``tol`` early-stopping
    check) and call :meth:`_gradient_descent` from their own ``fit``. See
    :mod:`cca_zoo._utils._ey` for the shared EY-loss machinery used by the
    CCA-family subclasses.

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means. Default True.
        learning_rate: Gradient step size. Default is 1e-2.
        max_iter: Number of gradient steps. Default is 1000.
        batch_size: Mini-batch size. ``None`` uses the full dataset.
        tol: Convergence tolerance on the objective change between
            consecutive steps. Default is 1e-6.
        momentum: Momentum coefficient in ``[0, 1)``. Default is 0.9.
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        learning_rate: float = 1e-2,
        max_iter: int = 1000,
        batch_size: int | None = None,
        tol: float = 1e-6,
        momentum: float = 0.9,
        random_state: int | None = None,
    ) -> None:
        super().__init__(latent_dimensions=latent_dimensions, center=center)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.tol = tol
        self.momentum = momentum
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
            views: Mini-batch of (possibly whitened) view arrays.
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
        """Scalar loss value, used only for the ``tol`` convergence check."""

    def _initial_weights(
        self, dims: list[int], rng: np.random.Generator
    ) -> list[np.ndarray]:
        """Random-orthogonal initial weights, one per view.

        Args:
            dims: Number of features per view.
            rng: Random generator.

        Returns:
            List of weight matrices, each (p_i, k) with orthonormal columns,
            where ``k = min(latent_dimensions, p_i)``.
        """
        weights = []
        for p in dims:
            k = min(self.latent_dimensions, p)
            w, _ = np.linalg.qr(rng.standard_normal((p, k)))
            weights.append(w)
        return weights

    def _gradient_descent(
        self, views: list[np.ndarray], rng: np.random.Generator
    ) -> list[np.ndarray]:
        """Run mini-batch momentum gradient descent to fit weight matrices.

        Args:
            views: List of arrays to fit on (raw or pre-whitened).
            rng: Random generator used for batching and initialisation.

        Returns:
            List of fitted weight matrices, one per view.
        """
        n = views[0].shape[0]
        bs = n if self.batch_size is None else min(self.batch_size, n)
        weights = self._initial_weights([v.shape[1] for v in views], rng)
        velocity = [np.zeros_like(w) for w in weights]
        prev_obj = np.inf
        for _ in range(self.max_iter):
            idx = rng.choice(n, bs, replace=False)
            batch = [v[idx] for v in views]
            representations = [b @ w for b, w in zip(batch, weights)]
            grads = self._derivative(batch, representations, weights)
            for i, g in enumerate(grads):
                velocity[i] = self.momentum * velocity[i] - self.learning_rate * g
                weights[i] = weights[i] + velocity[i]
            obj = self._objective(batch, representations, weights)
            if abs(prev_obj - obj) < self.tol:
                break
            prev_obj = obj
        return weights
