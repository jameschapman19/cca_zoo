"""Group-regularised CCA."""

from __future__ import annotations

from typing import Any, ClassVar, cast

import numpy as np
from numpy.typing import ArrayLike

from cca_zoo._utils._linalg import covariance
from cca_zoo._utils._param_constraints import NONNEGATIVE_PER_VIEW
from cca_zoo._utils._validation import perview_parameter
from cca_zoo.linear._mcca import MCCA


class GRCCA(MCCA):
    r"""MCCA with ridge penalties that shrink weights towards their group means.

    Tuzhilina et al.'s penalty on each view's weights $w$, for features in
    groups $g$ of size $p_g$ with mean weight $\bar w_g$,

    $$
    \sum_g \sum_{j \in g} (w_j - \bar w_g)^2 + \mu \sum_g p_g \bar w_g^2
        = w^\top \bigl((I - H) + \mu H\bigr) w,
    $$

    with $H$ the projection averaging each group, takes the place of
    :class:`MCCA`'s identity: each view's block of $B$ is
    $(1 - c)\Sigma_{ii} + c\,((I - H) + \mu H)$. ``shrinkage`` $c$ sets the
    strength of the penalty and ``mu`` that of the group means relative to
    the deviations from them: ``mu=1`` is MCCA's ridge, and ``mu=0`` shrinks
    each weight towards its group's mean alone. Their $\lambda$ and $\mu$
    are $c / (1 - c)$ and $\mu c / (1 - c)$.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.
        shrinkage: Strength of the penalty, in ``[0, 1]``: 0 is CCA. Per-view.
            Default is 0.
        mu: Penalty on the group means, relative to the within-group
            deviations. Per-view. Default is 0.
        feature_groups: Integer group label of each feature, shape
            (n_features_i,) per view; None puts each view in one group.
            Default is None.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).
        feature_groups_: Group label of each feature, per view.

    Raises:
        ValueError: If a view's ``feature_groups`` does not label each feature.

    References:
        Tuzhilina, E., Tozzi, L., & Hastie, T. (2023). Canonical correlation
        analysis in high dimensions with structured regularization.
        Statistical Modelling, 23(3), 203-227.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.linear import GRCCA
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 10))
        >>> X2 = rng.standard_normal((50, 8))
        >>> groups = [rng.integers(0, 3, size=10), rng.integers(0, 3, size=8)]
        >>> model = GRCCA(n_components=2, shrinkage=0.5, feature_groups=groups)
        >>> model = model.fit([X1, X2])
    """

    _supports_array_api: ClassVar[bool] = False
    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **{k: v for k, v in MCCA._parameter_constraints.items() if k != "pca"},
        "mu": NONNEGATIVE_PER_VIEW,
        "feature_groups": [None, list],
    }

    def __init__(
        self,
        n_components: int = 1,
        *,
        center: bool = True,
        shrinkage: float | list[float] = 0.0,
        mu: float | list[float] = 0.0,
        feature_groups: list[np.ndarray] | None = None,
    ) -> None:
        super().__init__(
            n_components=n_components,
            center=center,
            shrinkage=shrinkage,
            pca=False,
        )
        self.mu = mu
        self.feature_groups = feature_groups

    def fit(
        self,
        views: list[ArrayLike],
        y: None = None,
        sample_weight: ArrayLike | None = None,
    ) -> GRCCA:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.
            sample_weight: Weight of each sample; an integer weight is the
                same as repeating the sample. None weights samples equally.

        Returns:
            self.
        """
        return cast(GRCCA, super().fit(views, y, sample_weight))

    def _build_B(self, views: list[np.ndarray], c: list[float]) -> np.ndarray:
        """Block-diagonal covariance blended with each view's group penalty."""
        mu = perview_parameter("mu", self.mu, 0.0, len(views))
        groups = self.feature_groups or [np.zeros(v.shape[1], dtype=int) for v in views]
        for g, v in zip(groups, views):
            if len(g) != v.shape[1]:
                raise ValueError(
                    f"feature_groups has {len(g)} labels for a view with "
                    f"{v.shape[1]} features."
                )
        self.feature_groups_: list[np.ndarray] = [np.asarray(g) for g in groups]
        blocks = []
        for v, g, ci, mi in zip(views, self.feature_groups_, c, mu):
            same = (g[:, None] == g[None, :]).astype(float)
            averaging = same / same.sum(axis=1, keepdims=True)  # H
            penalty = np.eye(len(g)) - averaging + mi * averaging
            blocks.append((1.0 - ci) * covariance(v) + ci * penalty)
        B: np.ndarray = self._floored_blocks(blocks)
        return B
