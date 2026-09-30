"""Sparse alternating regression with BIC-selected sparsity (Wilms and Croux, 2015)."""

from __future__ import annotations

from typing import Any, ClassVar, cast

import numpy as np
from numpy.typing import ArrayLike
from sklearn.linear_model import lasso_path

from cca_zoo._base import BaseModel
from cca_zoo._utils._convergence import warn_if_not_converged
from cca_zoo._utils._param_constraints import POSITIVE_EPS, POSITIVE_INT
from cca_zoo.sparse._deflation import Deflation, others_score, ridge_cca_direction


class SAR(BaseModel):
    """Sparse alternating regression with BIC-selected sparsity.

    CCA as alternating regressions of each view's score on the other views'
    summed score, each a lasso whose penalty is chosen by BIC. Latent
    dimensions after the first are found on deflated views, then re-fitted
    against the original views (Wilms and Croux, Section 3). Each component
    starts from the ridge-regularised CCA direction, from which the first
    lasso finds a target the other views share. The extension beyond two
    views is this implementation's own.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.
        n_alphas: Penalties on each lasso path. Default is 100.
        max_iter: Maximum iterations per latent dimension. Default is 500.
        tol: Convergence tolerance of the alternating loop and each lasso path.
            Default is 1e-6.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).
        n_iter_: Most iterations run by any component.

    References:
        Wilms, I., & Croux, C. (2015). Sparse canonical correlation analysis
        from a predictive point of view. Biometrical Journal, 57(5), 834-851.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.sparse import SAR
        >>> rng = np.random.default_rng(0)
        >>> latent = rng.standard_normal(50)
        >>> X1 = np.column_stack([latent, rng.standard_normal((50, 9))])
        >>> X2 = np.column_stack([latent, rng.standard_normal((50, 7))])
        >>> model = SAR().fit([X1, X2])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "n_alphas": POSITIVE_INT,
        "max_iter": POSITIVE_INT,
        "tol": POSITIVE_EPS,
    }

    def __init__(
        self,
        n_components: int = 1,
        *,
        center: bool = True,
        n_alphas: int = 100,
        max_iter: int = 500,
        tol: float = 1e-6,
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.n_alphas = n_alphas
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, views: list[ArrayLike], y: None = None) -> SAR:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.

        Returns:
            self.
        """
        views_ = self._setup_fit(views)
        self.weights_: list[np.ndarray] = [
            np.zeros((p, self.n_components)) for p in self.n_features_per_view_
        ]
        self.n_iter_: int = 0
        converged = True
        deflation = Deflation(views_, self.n_components)
        for d, deflated in enumerate(deflation):
            w = ridge_cca_direction(deflated)
            for n_iter in range(1, self.max_iter + 1):
                previous = [wi.copy() for wi in w]
                # BIC-selected lasso of each view on the others' score.
                for i, view in enumerate(deflated):
                    w[i] = _unit_norm(
                        self._bic_lasso(view, others_score(deflated, w, i))
                    )
                if max(np.linalg.norm(a - b) for a, b in zip(w, previous)) < self.tol:
                    break
            else:
                converged = False
            self.n_iter_ = max(self.n_iter_, n_iter)
            # A later component's scores, re-fitted by lasso on the original
            # views, where the paper keeps its weights sparse.
            for i, (view, deflated_view, wi) in enumerate(zip(views_, deflated, w)):
                self.weights_[i][:, d] = (
                    wi
                    if d == 0
                    else _unit_norm(self._bic_lasso(view, deflated_view @ wi))
                )
            deflation.record(w)
        warn_if_not_converged(self, converged)
        self._fit_maps_and_importances(views_)
        return self

    def _bic_lasso(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Lasso coefficients at the BIC-minimising point of the lasso path.

        BIC is ``n log(RSS / n) + k log n``, with ``k`` the number of nonzero
        coefficients, as in Wilms and Croux (2015).
        """
        n, p = x.shape
        # A fit runs ~1000 paths, so skip sklearn's per-call input
        # validation, passing the Gram exactly when precompute="auto" would.
        x = np.asfortranarray(x, dtype=float)
        y = np.ascontiguousarray(y, dtype=float)
        xy = x.T @ y
        # sklearn's grid: from the smallest penalty zeroing every coefficient
        # down to 1e-3 of it, evenly on a log scale.
        alpha_max = max(float(np.abs(xy).max()) / n, np.finfo(float).resolution)
        _, coefs, _ = lasso_path(
            x,
            y,
            alphas=np.geomspace(alpha_max, 1e-3 * alpha_max, self.n_alphas),
            tol=self.tol,
            precompute=x.T @ x if n > p else False,
            Xy=xy if n > p else None,
            check_input=False,
        )
        rss = np.maximum(((y[:, None] - x @ coefs) ** 2).sum(axis=0), 1e-12)
        bic = n * np.log(rss / n) + (np.abs(coefs) > 1e-12).sum(axis=0) * np.log(n)
        return cast(np.ndarray, coefs[:, np.argmin(bic)])


def _unit_norm(w: np.ndarray) -> np.ndarray:
    """``w`` at unit norm; zero stays zero."""
    return w / max(float(np.linalg.norm(w)), 1e-12)
