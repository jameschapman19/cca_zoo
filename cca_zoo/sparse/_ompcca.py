r"""OrthogonalMatchingPursuitCCA — greedy fixed-cardinality sparse CCA, EY loss."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils._param_validation import Interval

from cca_zoo._base import BaseModel
from cca_zoo._utils._ey import omp_coordinate_descent_ey


class OrthogonalMatchingPursuitCCA(BaseModel):
    r"""OrthogonalMatchingPursuitCCA — fixed-cardinality CCA via greedy selection.

    Learns per-view linear weights $W_i$ with each view's number of active
    (nonzero-row) features capped at a fixed budget, the EY-loss analogue
    of sklearn's :class:`~sklearn.linear_model.OrthogonalMatchingPursuit`:
    where :class:`~cca_zoo.sparse.ElasticNetCCA` /
    :class:`~cca_zoo.sparse.MultiTaskElasticNetCCA` reach a sparsity level
    indirectly by tuning a continuous penalty strength, this fixes the
    sparsity level directly by specifying how many features each view may
    use.

    Fit by :func:`~cca_zoo._utils._ey.omp_coordinate_descent_ey`: for each
    view, features are added one at a time to a growing active set — each
    one chosen by the same criterion classical OMP uses (largest-magnitude
    gradient of the loss at a coefficient of zero, i.e. residual
    correlation, generalised here from a scalar to the norm of a $k$-vector
    since every feature has one coefficient per latent dimension) — and the
    active coefficients are re-solved to their *exact* joint optimum after
    every addition via :class:`~cca_zoo.sparse.ElasticNetCCA`'s own exact
    quartic coordinate solve with no penalty. Every feature outside a
    view's active set is exactly zero; nothing else is shrunk.

    Note:
        Like every EY-loss model, this is not convex, so different
        ``random_state`` initialisations (used only for the warm-started
        bootstrap the greedy search regrows from — see
        :func:`~cca_zoo._utils._ey.omp_coordinate_descent_ey`'s docstring
        for why a literal zero start doesn't work here) can select
        different active sets.

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means. Default is True.
        n_nonzero_coefs: Target number of active features per view. An
            ``int`` applies the same budget to every view; a list gives one
            budget per view (must match the number of views passed to
            ``fit``); ``None`` defaults each view to
            ``max(1, n_features_i // 10)``, mirroring sklearn's
            :class:`~sklearn.linear_model.OrthogonalMatchingPursuit`
            default. A budget larger than a view's own feature count is
            silently capped to that count. Default is None.
        max_iter: Maximum number of outer rounds cycling through every view
            and regrowing its active set from scratch. Default is 10.
        tol: Convergence tolerance on the (unpenalised) EY objective's
            change, both between outer rounds and between the
            coordinate-descent sweeps used to refit each active set.
            Default is 1e-6.
        random_state: Seed for the initial (dense, pre-selection) weights.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((200, 20))
        >>> X2 = rng.standard_normal((200, 15))
        >>> model = OrthogonalMatchingPursuitCCA(
        ...     latent_dimensions=2, n_nonzero_coefs=5
        ... ).fit([X1, X2])
        >>> scores = model.transform([X1, X2])
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
        n_nonzero_coefs: int | list[int] | None = None,
        max_iter: int = 10,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        super().__init__(latent_dimensions=latent_dimensions, center=center)
        self.n_nonzero_coefs = n_nonzero_coefs
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _resolve_n_nonzero_coefs(self, n_features: list[int]) -> list[int]:
        """Resolve ``n_nonzero_coefs`` into one positive budget per view.

        Args:
            n_features: Number of features in each view.

        Returns:
            One target active-feature count per view.

        Raises:
            ValueError: If a list is given with the wrong length, or any
                resolved budget is not a positive integer.
        """
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
        """Fit OrthogonalMatchingPursuitCCA by greedy forward selection on the EY loss.

        Args:
            views: List of 2 or more arrays, each (n_samples, n_features_i).
            y: Ignored.

        Returns:
            self: Fitted estimator.

        Raises:
            ValueError: If fewer than 2 views are provided.
            ValueError: If views have inconsistent numbers of samples.
            ValueError: If ``n_nonzero_coefs`` is a list of the wrong
                length, or resolves to a non-positive budget for some view.
        """
        views_ = self._setup_fit(views)
        n_nonzero_coefs = self._resolve_n_nonzero_coefs(self.n_features_in_)
        rng = np.random.default_rng(self.random_state)
        weights, _ = omp_coordinate_descent_ey(
            bases=views_,
            k=self.latent_dimensions,
            n_nonzero_coefs=n_nonzero_coefs,
            max_iter=self.max_iter,
            tol=self.tol,
            rng=rng,
        )
        self.weights_: list[np.ndarray] = weights
        return self
