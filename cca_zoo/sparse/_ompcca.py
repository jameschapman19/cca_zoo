"""Fixed-cardinality sparse CCA by greedy selection on the EY loss."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils._param_validation import Interval

from cca_zoo._base import BaseModel
from cca_zoo._utils._convergence import warn_if_not_converged
from cca_zoo._utils._ey import omp_coordinate_descent_ey
from cca_zoo._utils._param_constraints import POSITIVE_INT_PER_VIEW, RANDOM_STATE


class OrthogonalMatchingPursuitCCA(BaseModel):
    """Sparse multiview CCA with a fixed number of active features per view.

    The EY analogue of :class:`~sklearn.linear_model.OrthogonalMatchingPursuit`:
    each view's active set grows one feature at a time, choosing the feature
    with the largest EY gradient norm, and the active weights are refitted
    exactly after each addition. Features outside the active set are zero.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.
        n_nonzero_coefs: Active features, capped at the view's width; None
            is ``max(1, n_features_i // 10)``. Per-view. Default is None.
        max_iter: Maximum rounds of regrowing every view's active set.
            Default is 10.
        tol: Tolerance on the change in the EY loss. Default is 1e-6.
        random_state: Seed for the dense warm start. Default is None.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).
        n_iter_: Rounds of regrowing the active sets.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.sparse import OrthogonalMatchingPursuitCCA
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((200, 20))
        >>> X2 = rng.standard_normal((200, 15))
        >>> model = OrthogonalMatchingPursuitCCA(
        ...     n_components=2, n_nonzero_coefs=5
        ... ).fit([X1, X2])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="neither")],
        "n_nonzero_coefs": [None, *POSITIVE_INT_PER_VIEW],
        "random_state": RANDOM_STATE,
    }

    def __init__(
        self,
        n_components: int = 1,
        *,
        center: bool = True,
        n_nonzero_coefs: int | list[int] | None = None,
        max_iter: int = 10,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.n_nonzero_coefs = n_nonzero_coefs
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _resolve_n_nonzero_coefs(self, n_features: list[int]) -> list[int]:
        """One positive active-set size per view."""
        if self.n_nonzero_coefs is None:
            resolved = [max(1, p // 10) for p in n_features]
        elif isinstance(self.n_nonzero_coefs, (int, np.integer)):
            resolved = [int(self.n_nonzero_coefs)] * len(n_features)
        else:
            resolved = [int(x) for x in self.n_nonzero_coefs]
            if len(resolved) != len(n_features):
                raise ValueError(
                    f"n_nonzero_coefs has {len(resolved)} entries, expected "
                    f"one per view ({len(n_features)})."
                )
        if any(x < 1 for x in resolved):
            raise ValueError(
                f"n_nonzero_coefs must be positive for every view, got {resolved}."
            )
        return resolved

    def fit(
        self, views: list[ArrayLike], y: None = None
    ) -> OrthogonalMatchingPursuitCCA:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.

        Returns:
            self.

        Raises:
            ValueError: If ``n_nonzero_coefs`` has the wrong length or is not positive.
        """
        views_ = self._setup_fit(views)
        n_nonzero_coefs = self._resolve_n_nonzero_coefs(self.n_features_per_view_)
        rng = np.random.default_rng(self.random_state)
        weights, self.n_iter_, converged = omp_coordinate_descent_ey(
            bases=views_,
            k=self.n_components,
            n_nonzero_coefs=n_nonzero_coefs,
            max_iter=self.max_iter,
            tol=self.tol,
            rng=rng,
        )
        warn_if_not_converged(self, converged)
        self.weights_: list[np.ndarray] = weights
        return self._finish_fit(views_)
