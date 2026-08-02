"""GRCCA — Group Regularised Canonical Correlation Analysis."""

from __future__ import annotations

import warnings
from typing import cast

import numpy as np
from numpy.typing import ArrayLike

from cca_zoo._utils._linalg import gevp
from cca_zoo._utils._validation import perview_parameter
from cca_zoo.linear._mcca import MCCA


class GRCCA(MCCA):
    r"""Group Regularised Canonical Correlation Analysis.

    Extends :class:`MCCA` with structured ridge regularisation that shrinks
    within-group feature weights toward a shared group-level effect. Each
    view's features are partitioned into groups via ``feature_groups``; the
    per-view ``c`` parameter controls shrinkage of within-group deviations
    and ``mu`` controls the weighting of the group-level effect.

    Each view is internally augmented with group-mean features before
    solving the generalised eigenvalue problem, then the resulting weights
    are algebraically collapsed back to the original feature space, so
    ``transform`` operates directly on the un-augmented views.

    References:
        Tuzhilina, E., Tozzi, L., & Hastie, T. (2021). Canonical correlation
        analysis in high dimensions with structured regularization.
        *Statistical Modelling*.

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means before fitting. Default True.
        c: Ridge regularisation parameter(s) controlling within-group
            shrinkage. Either a scalar applied to all views or a per-view
            list, each in ``[0, 1]``. ``c=0`` disables grouping for that
            view (falls back to plain MCCA behaviour). Default is 0.
        mu: Regularisation parameter(s) controlling the group-level effect
            scale. Either a scalar or a per-view list. Default is 0.
        eps: Small constant added to the eigenvalues of B to ensure
            positive definiteness. Default is 1e-6.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 10))
        >>> X2 = rng.standard_normal((50, 8))
        >>> groups1 = rng.integers(0, 3, size=10)
        >>> groups2 = rng.integers(0, 3, size=8)
        >>> model = GRCCA(latent_dimensions=2, c=0.5).fit(
        ...     [X1, X2], feature_groups=[groups1, groups2]
        ... )
        >>> scores = model.transform([X1, X2])
    """

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        c: float | list[float] = 0.0,
        mu: float | list[float] = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(
            latent_dimensions=latent_dimensions,
            center=center,
            c=c,
            pca=False,
            eps=eps,
        )
        self.mu = mu

    def fit(
        self,
        views: list[ArrayLike],
        y: None = None,
        feature_groups: list[np.ndarray] | None = None,
    ) -> GRCCA:
        """Fit the GRCCA model.

        Args:
            views: List of arrays, each (n_samples, n_features_i).
            y: Ignored.
            feature_groups: List of integer group-label arrays, one per
                view, each of shape (n_features_i,). Required for
                meaningful grouping whenever the corresponding per-view
                ``c`` is nonzero; defaults to a single group per view
                (equivalent to plain MCCA) if omitted.

        Returns:
            self: Fitted estimator.
        """
        views_ = self._setup_fit(views)
        c_ = perview_parameter("c", self.c, 0.0, self.n_views_)
        mu_ = perview_parameter("mu", self.mu, 0.0, self.n_views_)

        if feature_groups is None:
            if any(ci > 0 for ci in c_):
                warnings.warn(
                    "No feature_groups provided; using a single group per "
                    "view, which makes the group regularisation a no-op."
                )
            feature_groups = [np.ones(v.shape[1], dtype=int) for v in views_]
        self.feature_groups_ = feature_groups

        processed = [
            self._augment_view(v, g, m, c)
            for v, g, m, c in zip(views_, feature_groups, mu_, c_)
        ]
        A = self._build_A(processed)
        B = self._build_B(processed, c_)
        _, eigvecs = gevp(A, B, self.latent_dimensions)
        splits = np.cumsum([v.shape[1] for v in processed])
        raw_blocks = np.split(eigvecs, splits[:-1], axis=0)

        self.weights_: list[np.ndarray] = [
            self._collapse_weights(block, g, c, m)
            for block, g, c, m in zip(raw_blocks, feature_groups, c_, mu_)
        ]
        return self

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _augment_view(
        self, view: np.ndarray, group: np.ndarray, mu: float, c: float
    ) -> np.ndarray:
        """Augment a view with per-group mean features scaled by c and mu."""
        if c <= 0:
            return view
        _, unique_inverse, unique_counts, group_means = self._group_mean(view, group)
        mu_eff = 1.0 if mu == 0 else mu
        view_1 = (view - group_means[:, unique_inverse]) / c
        view_2 = group_means / np.sqrt(mu_eff / unique_counts)
        return np.hstack((view_1, view_2))

    def _collapse_weights(
        self,
        block: np.ndarray,
        group: np.ndarray,
        c: float,
        mu: float,
    ) -> np.ndarray:
        """Collapse augmented-space eigenvectors back to the original feature space."""
        if c <= 0:
            return block
        n_groups = np.unique(group).shape[0]
        weights_1 = block[:-n_groups]
        weights_2 = block[-n_groups:]
        _, unique_inverse, unique_counts, group_means = self._group_mean(
            weights_1.T, group
        )
        mu_eff = 1.0 if mu == 0 else mu
        weights_1 = (weights_1 - group_means[:, unique_inverse].T) / c
        weights_2 = weights_2 / np.sqrt(mu_eff * unique_counts[:, None])
        return cast(np.ndarray, weights_1 + weights_2[unique_inverse])

    @staticmethod
    def _group_mean(
        arr: np.ndarray, group: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute the per-group mean along the column axis of ``arr``.

        Args:
            arr: Array of shape (n_rows, n_features).
            group: Integer group label for each column, shape (n_features,).

        Returns:
            Tuple ``(ids, unique_inverse, unique_counts, group_means)``
            where ``group_means`` has shape (n_rows, n_groups).
        """
        ids, unique_inverse, unique_counts = np.unique(
            group, return_inverse=True, return_counts=True
        )
        group_means = np.array([arr[:, group == g].mean(axis=1) for g in ids]).T
        return ids, unique_inverse, unique_counts, group_means
