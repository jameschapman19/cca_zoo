"""Eckart-Young CCA by mini-batch stochastic gradient descent."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils import gen_batches
from sklearn.utils._param_validation import Interval
from sklearn.utils.extmath import randomized_svd

from cca_zoo._utils._convergence import warn_if_not_converged
from cca_zoo._utils._ey import cheap_orthonormal_projection_weights
from cca_zoo.linear.gradient._cca_ey import CCAEY


class StochasticCCAEY(CCAEY):
    """Multiview CCA by mini-batch SGD on the Eckart-Young loss.

    The loss of :class:`~cca_zoo.linear.gradient.CCAEY`, minimised as
    :class:`~sklearn.linear_model.SGDRegressor` does with
    ``learning_rate="adaptive"``: each epoch shuffles the data and takes one
    step per mini-batch, and whenever the epoch loss stalls the step is
    divided by 5, so that the fit settles rather than hovering at the noise
    of its mini-batches. For data too large for full-batch gradients.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.
        shrinkage: Shrinkage of each view's covariance towards the identity,
            in ``[0, 1]``: 0 is CCA and 1 is PLS. Default is 0.
        learning_rate: Step size as a fraction of ``1 / L_i`` for view i,
            where ``L_i`` is the largest eigenvalue of its constraint matrix
            ``(1 - shrinkage) cov_i + shrinkage I``, so that the default suits
            views in any units. The initial step. Default is 0.2.
        batch_size: Mini-batch size; None uses all samples. Default is None.
        max_iter: Number of epochs. Default is 1000.
        n_iter_no_change: Epochs without an improvement of ``tol`` on the
            best epoch loss, the mean of its mini-batches' losses, before the
            step is divided by 5; the fit stops once the step is below 1e-6.
            Default is 5, as in sklearn's SGD.
        tol: Improvement in the epoch loss that counts, as for
            ``n_iter_no_change``. Default is 1e-6.
        random_state: Seed for the shuffling and initial weights. Default is
            None.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).
        n_iter_: Epochs run.

    References:
        Chapman, J., Wells, L., & Lawry Aguila, A. (2024). Unconstrained
        Stochastic CCA: Unifying Multiview and Self-Supervised Learning.
        arXiv:2310.01012.

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
        "batch_size": [None, Interval(Integral, 2, None, closed="left")],
        "n_iter_no_change": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(
        self,
        n_components: int = 1,
        *,
        center: bool = True,
        shrinkage: float = 0.0,
        learning_rate: float = 0.2,
        batch_size: int | None = None,
        max_iter: int = 1000,
        n_iter_no_change: int = 5,
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
        self.batch_size = batch_size
        self.n_iter_no_change = n_iter_no_change

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
        self._fit_maps_and_importances(views_)
        return self

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
        """Weights from mini-batch SGD with sklearn's adaptive step."""
        n = views[0].shape[0]
        bs = n if self.batch_size is None else min(self.batch_size, n)
        # Each view's step is set by the curvature a mini-batch sees, which
        # exceeds the full data's when the batch is small next to the view.
        rows = rng.choice(n, bs, replace=False)
        c = self.shrinkage
        variances = [
            randomized_svd(v[rows], 1, random_state=0)[1][0] ** 2 / (bs - 1)
            for v in views
        ]
        constraints = [(1 - c) * s + c for s in variances]
        # The loss's curvature is the constraint's times the largest canonical
        # value, which is at most 1 for CCA but a covariance for PLS; the
        # variance over the constraint bounds it, 1 at shrinkage 0.
        canonical = max(s / t for s, t in zip(variances, constraints))
        scales = [1.0 / (t * canonical) for t in constraints]
        weights = self._initial_weights(views, rng)
        eta, best_obj, stalled = self.learning_rate, np.inf, 0
        converged = False
        for n_iter in range(1, self.max_iter + 1):
            perm = rng.permutation(n)
            shuffled = [v[perm] for v in views]
            # The epoch's loss is the mean of its mini-batches' losses, as in
            # sklearn's SGD, so the fit never needs a pass over all the data.
            # The remainder joins the last full batch: a batch of one row has
            # no covariance.
            obj = 0.0
            for sl in gen_batches(n, bs, min_batch_size=bs):
                batch = [v[sl] for v in shuffled]
                representations = [b @ w for b, w in zip(batch, weights)]
                obj += (
                    self._objective(batch, representations, weights) * len(batch[0]) / n
                )
                grads = self._derivative(batch, representations, weights)
                weights = [w - eta * s * g for w, s, g in zip(weights, scales, grads)]
            if not np.isfinite(obj):
                raise ValueError(
                    "StochasticCCAEY diverged. Lower learning_rate, or scale the "
                    "views with StandardScaler."
                )
            stalled = stalled + 1 if obj > best_obj - self.tol else 0
            best_obj = min(best_obj, obj)
            # As sklearn's SGD with learning_rate="adaptive": a stall divides
            # the step by 5, and a stall at a negligible step is convergence.
            if stalled >= self.n_iter_no_change:
                if eta <= 1e-6:
                    converged = True
                    break
                eta, stalled = eta / 5, 0
        self.n_iter_: int = n_iter
        warn_if_not_converged(self, converged)
        return weights
