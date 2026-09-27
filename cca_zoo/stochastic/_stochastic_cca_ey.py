"""Eckart-Young CCA by mini-batch stochastic gradient descent."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils import gen_batches
from sklearn.utils._param_validation import Interval

from cca_zoo._utils._convergence import warn_if_not_converged
from cca_zoo._utils._ey import cheap_orthonormal_projection_weights
from cca_zoo.linear.gradient._cca_ey import CCAEY


class StochasticCCAEY(CCAEY):
    """Multiview CCA by mini-batch momentum SGD on the Eckart-Young loss.

    The loss of :class:`~cca_zoo.linear.gradient.CCAEY`, minimised as
    :class:`~sklearn.linear_model.SGDRegressor` does: each epoch shuffles the
    data and takes one momentum step per mini-batch. For data too large for
    full-batch gradients.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.
        shrinkage: Shrinkage of each view's covariance towards the identity,
            in ``[0, 1]``: 0 is CCA and 1 is PLS. Default is 0.
        learning_rate: Step size. Default is 1e-2.
        momentum: Momentum in ``[0, 1)``. Default is 0.9.
        batch_size: Mini-batch size; None uses all samples. Default is None.
        max_iter: Number of epochs. Default is 1000.
        tol: Tolerance on the change in the full-data loss between epochs.
            Default is 1e-6.
        random_state: Seed for the shuffling and initial weights. Default is
            None.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).
        n_iter_: Epochs run.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.stochastic import StochasticCCAEY
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((5000, 200))
        >>> X2 = rng.standard_normal((5000, 150))
        >>> model = StochasticCCAEY(n_components=4, batch_size=128, random_state=0)
        >>> model = model.fit([X1, X2])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **CCAEY._parameter_constraints,
        "learning_rate": [Interval(Real, 0, None, closed="neither")],
        "momentum": [Interval(Real, 0, 1, closed="left")],
        "batch_size": [None, Interval(Integral, 1, None, closed="left")],
    }

    def __init__(
        self,
        n_components: int = 1,
        center: bool = True,
        shrinkage: float = 0.0,
        learning_rate: float = 1e-2,
        momentum: float = 0.9,
        batch_size: int | None = None,
        max_iter: int = 1000,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            n_components=n_components,
            center=center,
            shrinkage=shrinkage,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size

    def fit(self, views: list[ArrayLike], y: None = None) -> StochasticCCAEY:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.

        Returns:
            self.

        Raises:
            ValueError: If the updates diverge.
        """
        views_: list[np.ndarray] = self._setup_fit(views)
        rng = np.random.default_rng(self.random_state)
        self.weights_ = self._fit_sgd(views_, rng)
        return self._finish_fit(views_)

    def _initial_weights(
        self, views: list[np.ndarray], rng: np.random.Generator
    ) -> list[np.ndarray]:
        """As ``CCAEY``'s, but on one mini-batch."""
        return cheap_orthonormal_projection_weights(
            views, self.n_components, self.batch_size, rng
        )

    def _fit_sgd(
        self, views: list[np.ndarray], rng: np.random.Generator
    ) -> list[np.ndarray]:
        """Weights from mini-batch momentum SGD."""
        n = views[0].shape[0]
        bs = n if self.batch_size is None else min(self.batch_size, n)
        weights = self._initial_weights(views, rng)
        velocity = [np.zeros_like(w) for w in weights]
        prev_obj = np.inf
        converged = False
        for n_iter in range(1, self.max_iter + 1):
            perm = rng.permutation(n)
            shuffled = [v[perm] for v in views]
            for sl in gen_batches(n, bs):
                batch = [v[sl] for v in shuffled]
                representations = [b @ w for b, w in zip(batch, weights)]
                grads = self._derivative(batch, representations, weights)
                for i, g in enumerate(grads):
                    velocity[i] = self.momentum * velocity[i] - self.learning_rate * g
                    weights[i] = weights[i] + velocity[i]
            full_representations = [v @ w for v, w in zip(views, weights)]
            obj = self._objective(views, full_representations, weights)
            if not np.isfinite(obj):
                raise ValueError(
                    "StochasticCCAEY diverged. Lower learning_rate, or scale the "
                    "views with StandardScaler."
                )
            if abs(prev_obj - obj) < self.tol:
                converged = True
                break
            prev_obj = obj
        self.n_iter_: int = n_iter
        warn_if_not_converged(self, converged)
        return weights
