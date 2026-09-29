"""Sparse CCA by hard-thresholded alternating updates, after Asteris et al. (2016)."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike

from cca_zoo._base import BaseModel
from cca_zoo._utils._convergence import warn_if_not_converged
from cca_zoo._utils._param_constraints import (
    POSITIVE_EPS,
    POSITIVE_INT,
    POSITIVE_INT_PER_VIEW,
    RANDOM_STATE,
)
from cca_zoo._utils._validation import perview_parameter
from cca_zoo.sparse._deflation import Deflation, others_score, pls_direction


class SpanCCA(BaseModel):
    """Sparse CCA by hard-thresholded alternating updates.

    Each update keeps the ``span`` largest-magnitude weights. This is an
    alternating heuristic in the spirit of Asteris et al.'s SpanCCA, not their
    randomised low-rank search.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means. Default is True.
        span: Number of nonzero weights kept; ``None`` keeps all. Per-view.
            Default is None.
        max_iter: Maximum iterations per latent dimension. Default is 500.
        tol: Convergence tolerance on the change in weights. Default is 1e-6.
        random_state: Seed for the start. Default is None.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).
        n_iter_: Most iterations run by any component.

    References:
        Asteris, M., Kyrillidis, A., Koyejo, O., & Poldrack, R. (2016). A simple
        and provable algorithm for sparse diagonal CCA. ICML.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.sparse import SpanCCA
        >>> rng = np.random.default_rng(0)
        >>> X1, X2 = rng.standard_normal((50, 10)), rng.standard_normal((50, 8))
        >>> model = SpanCCA(span=5, random_state=0).fit([X1, X2])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "span": [None, *POSITIVE_INT_PER_VIEW],
        "max_iter": POSITIVE_INT,
        "tol": POSITIVE_EPS,
        "random_state": RANDOM_STATE,
    }

    def __init__(
        self,
        n_components: int = 1,
        *,
        center: bool = True,
        span: int | list[int] | None = None,
        max_iter: int = 500,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.span = span
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, views: list[ArrayLike], y: None = None) -> SpanCCA:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.

        Returns:
            self.
        """
        views_ = self._setup_fit(views)
        rng = np.random.default_rng(self.random_state)
        spans = (
            self.n_features_per_view_
            if self.span is None
            else [
                int(s) for s in perview_parameter("span", self.span, 1, self.n_views_)
            ]
        )
        self.n_iter_: int = 0
        converged = True
        deflation = Deflation(views_, self.n_components)
        for deflated in deflation:
            w = pls_direction(deflated, rng)
            for n_iter in range(1, self.max_iter + 1):
                previous = [wi.copy() for wi in w]
                # Power iteration keeping each view's `span` largest weights.
                for i, (view, span) in enumerate(zip(deflated, spans)):
                    raw = view.T @ others_score(deflated, w, i)
                    if span < len(raw):
                        raw[np.abs(raw) < np.sort(np.abs(raw))[-span]] = 0.0
                    w[i] = raw / max(np.linalg.norm(raw), 1e-12)
                if max(np.linalg.norm(a - b) for a, b in zip(w, previous)) < self.tol:
                    break
            else:
                converged = False
            self.n_iter_ = max(self.n_iter_, n_iter)
            deflation.record(w)
        warn_if_not_converged(self, converged)
        self.weights_: list[np.ndarray] = deflation.weights()
        self._fit_maps_and_importances(views_)
        return self
