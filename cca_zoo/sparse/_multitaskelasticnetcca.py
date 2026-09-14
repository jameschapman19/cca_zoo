r"""MultiTaskElasticNetCCA — row-group-sparse linear CCA on the EY loss."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils._param_validation import Interval

from cca_zoo._base import BaseModel
from cca_zoo._utils._ey import group_coordinate_descent_ey


class MultiTaskElasticNetCCA(BaseModel):
    r"""MultiTaskElasticNetCCA — row-group-sparse linear CCA, fit on the EY loss.

    Like :class:`~cca_zoo.sparse.ElasticNetCCA`, learns per-view linear
    weights $W_i$ minimising an elastic-net-penalised EY loss, but with
    sklearn's :class:`~sklearn.linear_model.MultiTaskLasso` /
    :class:`~sklearn.linear_model.MultiTaskElasticNet` row-group penalty in
    place of a plain per-scalar penalty:

    $$
    \mathcal{L}(W) = \mathcal{L}_{EY}(Z_1, \dots, Z_M)
        + \sum_i \left( \alpha \, \rho \, \|W_i\|_{2,1}
        + \tfrac{1}{2} \alpha (1-\rho) \|W_i\|_F^2 \right)
    $$

    where $\|W_i\|_{2,1} = \sum_j \|W_i[j,:]\|_2$ sums each *feature's*
    weight-row Euclidean norm over all ``latent_dimensions`` components.
    Because the penalty on a row is zero only when the whole row is zero,
    a feature is either used by every canonical variate or by none —
    unlike :class:`~cca_zoo.sparse.ElasticNetCCA`, which can (and often
    does) keep a feature for component 1 while dropping it from component
    2. That joint selection is exactly what
    :class:`~sklearn.linear_model.MultiTaskLasso` buys over plain
    :class:`~sklearn.linear_model.Lasso` for ordinary multi-output
    regression, and it is arguably an even more natural fit here, since a
    CCA model's ``latent_dimensions`` are not independent "tasks" to be
    fit separately but different views of the same underlying features.

    Fit by :func:`~cca_zoo._utils._ey.group_coordinate_descent_ey` —
    block-coordinate descent, one *row* (feature, across all components) at
    a time, using proximal gradient (ISTA) with backtracking line search
    rather than :class:`~cca_zoo.sparse.ElasticNetCCA`'s exact per-scalar
    quartic solve. The two aren't interchangeable: a row's smooth EY
    restriction couples all $k$ of its entries together (through the
    auto-covariance's off-diagonal terms), so there is no closed form for
    the exact row minimiser the way there is for a single scalar — see that
    function's docstring for the full derivation. Every accepted step is
    still a verified decrease of the exact penalised objective, so fitting
    remains provably monotonic, just without per-step exactness.

    Note:
        Like every EY-loss model, this is not convex in $W$, so different
        ``random_state`` initialisations can land on different stationary
        points — see :class:`~cca_zoo.sparse.ElasticNetCCA`'s docstring.

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means. Default is True.
        alpha: Overall penalty strength. Default is 1.0.
        l1_ratio: Mixing parameter in ``[0, 1]``; 0 is pure (Frobenius)
            ridge, 1 is pure row-group lasso. Default is 0.5.
        max_iter: Maximum number of full coordinate-descent sweeps.
            Default is 100.
        tol: Convergence tolerance on the penalised objective's change
            between consecutive sweeps. Default is 1e-6.
        random_state: Seed for the initial weights.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((200, 20))
        >>> X2 = rng.standard_normal((200, 15))
        >>> model = MultiTaskElasticNetCCA(latent_dimensions=2, alpha=0.1).fit([X1, X2])
        >>> scores = model.transform([X1, X2])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "alpha": [Interval(Real, 0, None, closed="left")],
        "l1_ratio": [Interval(Real, 0, 1, closed="both")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="neither")],
    }

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        max_iter: int = 100,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        super().__init__(latent_dimensions=latent_dimensions, center=center)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, views: list[ArrayLike], y: None = None) -> MultiTaskElasticNetCCA:
        """Fit MultiTaskElasticNetCCA by row-group coordinate descent on the EY loss.

        Args:
            views: List of 2 or more arrays, each (n_samples, n_features_i).
            y: Ignored.

        Returns:
            self: Fitted estimator.

        Raises:
            ValueError: If fewer than 2 views are provided.
            ValueError: If views have inconsistent numbers of samples.
        """
        views_ = self._setup_fit(views)
        rng = np.random.default_rng(self.random_state)
        weights, _ = group_coordinate_descent_ey(
            bases=views_,
            k=self.latent_dimensions,
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            max_iter=self.max_iter,
            tol=self.tol,
            rng=rng,
        )
        self.weights_: list[np.ndarray] = weights
        return self
