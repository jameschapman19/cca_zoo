"""Sparse CCA by linearised ADMM (Suo et al., 2017)."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike

from cca_zoo._base import BaseModel
from cca_zoo._utils._convergence import warn_if_not_converged
from cca_zoo._utils._linalg import soft_threshold
from cca_zoo._utils._param_constraints import (
    NONNEGATIVE_PER_VIEW,
    POSITIVE_EPS,
    POSITIVE_INT,
    RANDOM_STATE,
)
from cca_zoo._utils._validation import perview_parameter
from cca_zoo.sparse._deflation import Deflation, pls_direction


class ADMMCCA(BaseModel):
    r"""Sparse CCA by linearised ADMM.

    For view $i$, with the other views' weights fixed, solves

    $$
    \max_{\mathbf{w}_i}\ \tfrac{1}{n} \mathbf{w}_i^\top X_i^\top
        \textstyle\sum_{j \ne i} X_j \mathbf{w}_j
        - \alpha_i \|\mathbf{w}_i\|_1
    \quad\text{subject to}\quad \tfrac{1}{n} \|X_i \mathbf{w}_i\|_2^2 \le 1,
    $$

    the paper's problem on $X_i / \sqrt{n}$, so that the constraint is unit
    variance and ``alpha`` means the same at any sample size. It is solved
    by linearised ADMM on the split $\mathbf{z}_i = X_i \mathbf{w}_i / \sqrt{n}$,
    so each step is a soft-threshold and a projection onto the unit ball,
    with step size $n / (\rho \|X_i\|_{\mathrm{op}}^2)$. Further components
    use deflation rather than the paper's orthogonality constraint.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means. Default is True.
        alpha: L1 penalty. Per-view. Default is 0.1.
        rho: ADMM augmented-Lagrangian penalty. Default is 1.0.
        max_iter: Maximum outer iterations per latent dimension. Default is 500.
        admm_iter: Maximum ADMM iterations per view update. Default is 50.
        tol: Convergence tolerance of both loops. Default is 1e-6.
        random_state: Seed for the start. Default is None.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).
        n_iter_: Most iterations run by any component.

    References:
        Suo, X., Mineiro, P., & Anandkumar, A. (2017). Sparse canonical
        correlation analysis. arXiv:1705.10865.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.sparse import ADMMCCA
        >>> rng = np.random.default_rng(0)
        >>> X1, X2 = rng.standard_normal((50, 10)), rng.standard_normal((50, 8))
        >>> model = ADMMCCA(alpha=0.1, random_state=0).fit([X1, X2])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "alpha": NONNEGATIVE_PER_VIEW,
        "rho": POSITIVE_EPS,
        "admm_iter": POSITIVE_INT,
        "max_iter": POSITIVE_INT,
        "tol": POSITIVE_EPS,
        "random_state": RANDOM_STATE,
    }

    def __init__(
        self,
        n_components: int = 1,
        *,
        center: bool = True,
        alpha: float | list[float] = 0.1,
        rho: float = 1.0,
        max_iter: int = 500,
        admm_iter: int = 50,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.alpha = alpha
        self.rho = rho
        self.max_iter = max_iter
        self.admm_iter = admm_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, views: list[ArrayLike], y: None = None) -> ADMMCCA:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.

        Returns:
            self.
        """
        views_ = self._setup_fit(views)
        rng = np.random.default_rng(self.random_state)
        alphas = perview_parameter("alpha", self.alpha, 0.1, self.n_views_)
        self.n_iter_: int = 0
        converged = True
        deflation = Deflation(views_, self.n_components)
        for deflated in deflation:
            w = pls_direction(deflated, rng)
            # The paper's problem on X / sqrt(n): unit-variance constraints.
            scaled = [v / np.sqrt(len(v)) for v in deflated]
            # The split z_i = X_i w_i and its scaled dual persist across the
            # outer iterations, as in the paper's Algorithm 1.
            z = [v @ wi for v, wi in zip(scaled, w)]
            dual = [np.zeros(len(v)) for v in scaled]
            steps = [1.0 / (self.rho * np.linalg.norm(v, ord=2) ** 2) for v in scaled]
            for n_iter in range(1, self.max_iter + 1):
                previous = [wi.copy() for wi in w]
                for i, (view, alpha, step) in enumerate(zip(scaled, alphas, steps)):
                    others = sum(
                        v @ wj for j, (v, wj) in enumerate(zip(scaled, w)) if j != i
                    )
                    w[i], z[i], dual[i] = self._admm_block(
                        view, view.T @ others, alpha, step, w[i], z[i], dual[i]
                    )
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

    def _admm_block(
        self,
        view: np.ndarray,
        reward: np.ndarray,
        alpha: float,
        step: float,
        w: np.ndarray,
        z: np.ndarray,
        dual: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One view's block: max ``w' reward - alpha ||w||_1`` s.t. ``||X w|| <= 1``.

        Linearised ADMM on the split ``z = X w``: a soft-thresholded gradient
        step in ``w``, a projection of ``z`` onto the unit ball, and a dual
        ascent step, with ``step`` ``1 / (rho ||X||_op^2)``.

        Returns:
            The updated ``(w, z, dual)``.
        """
        for _ in range(self.admm_iter):
            w_before = w
            gradient = self.rho * view.T @ (view @ w - z + dual) - reward
            w = soft_threshold(w - step * gradient, step * alpha)
            score = view @ w
            z = (score + dual) / max(1.0, np.linalg.norm(score + dual))
            dual = dual + score - z
            if np.linalg.norm(w - w_before) < self.tol:
                break
        return w, z, dual
