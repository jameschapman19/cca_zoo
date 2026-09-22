"""StochasticCCAEY — mini-batch momentum SGD on the EY loss."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils import gen_batches
from sklearn.utils._param_validation import Interval

from cca_zoo._utils._ey import cheap_orthonormal_projection_weights
from cca_zoo.linear.gradient._cca_ey import CCAEY


class StochasticCCAEY(CCAEY):
    r"""Eckart-Young CCA fit by mini-batch momentum SGD, for large-scale data.

    Identical objective to :class:`~cca_zoo.linear.gradient.CCAEY` (same
    ``c`` ridge blend towards :class:`~cca_zoo.linear.gradient.PLSEY`, same
    analytic gradient) but fit the way
    :class:`~sklearn.linear_model.SGDRegressor` fits a linear model: each
    epoch, the data is shuffled once and split into ``batch_size`` chunks
    (:func:`sklearn.utils.gen_batches`), taking one momentum gradient step
    per chunk. Use this instead of :class:`~cca_zoo.linear.gradient.CCAEY`
    when the full dataset does not fit comfortably in memory or a full-batch
    gradient evaluation is too slow to repeat every iteration.

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means. Default True.
        c: Ridge blend in ``[0, 1]`` between ``CCAEY`` (0) and ``PLSEY``
            (1). Default is 0; see :class:`~cca_zoo.linear.gradient.CCAEY`'s
            docstring for the numerical-stability note on high-dimensional
            data.
        learning_rate: Gradient step size. Default is 1e-2.
        momentum: Momentum coefficient in ``[0, 1)``. Default is 0.9.
        batch_size: Mini-batch size. ``None`` uses the full dataset (one
            gradient step per epoch).
        max_iter: Number of epochs (full passes over the shuffled data).
            Default is 1000.
        tol: Convergence tolerance on the full-dataset objective's change
            between consecutive epochs. Default is 1e-6.
        random_state: Seed for reproducibility.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((5000, 200))
        >>> X2 = rng.standard_normal((5000, 150))
        >>> model = StochasticCCAEY(latent_dimensions=4, batch_size=128, random_state=0)
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
        latent_dimensions: int = 1,
        center: bool = True,
        c: float = 0.0,
        learning_rate: float = 1e-2,
        momentum: float = 0.9,
        batch_size: int | None = None,
        max_iter: int = 1000,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            latent_dimensions=latent_dimensions,
            center=center,
            c=c,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size

    def fit(self, views: list[ArrayLike], y: None = None) -> StochasticCCAEY:
        """Fit by mini-batch momentum SGD on the EY loss.

        Args:
            views: List of 2 or more arrays, each (n_samples, n_features_i).
            y: Ignored.

        Returns:
            self: Fitted estimator.

        Raises:
            ValueError: If fewer than 2 views are provided.
            ValueError: If views have inconsistent numbers of samples.
        """
        views_: list[np.ndarray] = self._setup_fit(views)
        rng = np.random.default_rng(self.random_state)
        self.weights_ = self._fit_sgd(views_, rng)
        return self

    def _initial_weights(
        self, views: list[np.ndarray], rng: np.random.Generator
    ) -> list[np.ndarray]:
        """Cheap, data-informed initial weights, on one mini-batch.

        As :meth:`~cca_zoo.linear.gradient.CCAEY._initial_weights`, but
        projected on a single ``batch_size`` mini-batch rather than the
        full dataset, since a full-batch pass is exactly what this class
        exists to avoid.
        """
        return cheap_orthonormal_projection_weights(
            views, self.latent_dimensions, self.batch_size, rng
        )

    def _fit_sgd(
        self, views: list[np.ndarray], rng: np.random.Generator
    ) -> list[np.ndarray]:
        """Run mini-batch momentum SGD to fit weight matrices.

        Args:
            views: List of arrays to fit on.
            rng: Random generator used for shuffling and initialisation.

        Returns:
            List of fitted weight matrices, one per view.
        """
        n = views[0].shape[0]
        bs = n if self.batch_size is None else min(self.batch_size, n)
        weights = self._initial_weights(views, rng)
        velocity = [np.zeros_like(w) for w in weights]
        prev_obj = np.inf
        for _ in range(self.max_iter):
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
            if abs(prev_obj - obj) < self.tol:
                break
            prev_obj = obj
        return weights
