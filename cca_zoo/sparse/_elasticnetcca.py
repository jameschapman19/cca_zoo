r"""ElasticNetCCA — sparse linear CCA via coordinate descent directly on the EY loss."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils._param_validation import Interval

from cca_zoo._base import BaseModel
from cca_zoo._utils._ey import coordinate_descent_ey


class ElasticNetCCA(BaseModel):
    r"""ElasticNetCCA — sparse linear CCA by coordinate descent on the EY loss.

    Learns per-view linear weights $W_i$ (embeddings $Z_i = X_i W_i$) that
    minimise the elastic-net-penalised Eckart-Young (EY) objective:

    $$
    \mathcal{L}(W) = \mathcal{L}_{EY}(Z_1, \dots, Z_M)
        + \sum_i \left( \alpha \, \rho \, \|W_i\|_1
        + \tfrac{1}{2} \alpha (1-\rho) \|W_i\|_F^2 \right)
    $$

    where $\mathcal{L}_{EY}$ is the EY loss (see
    :mod:`cca_zoo._utils._ey`, shared with
    :class:`~cca_zoo.linear.gradient.CCAEY`, :class:`~cca_zoo.tree.TreeCCA`,
    :class:`~cca_zoo.gam.GAMCCA`, and :class:`~cca_zoo.gp.GaussianProcessCCA`)
    and $\rho$ is ``l1_ratio``. This is fit by
    :func:`~cca_zoo._utils._ey.coordinate_descent_ey` — cyclic coordinate
    descent on the EY loss itself (``bases`` = the raw centred views, no
    ridge coupling), the same algorithm
    :class:`~sklearn.linear_model.ElasticNet` uses for ordinary
    (squared-error) elastic net, but with each coordinate's exact minimiser
    solved against $\mathcal{L}_{EY}$'s own (quartic, not quadratic)
    restriction — see that function's docstring for the derivation.

    Because every update is the *exact* per-coordinate minimiser of the
    *exact* (not linearised or diagonal-Hessian-approximated) EY loss, this
    needs no post-hoc whitening/decorrelation step of the kind
    :class:`~cca_zoo.gam.GAMCCA` and :class:`~cca_zoo.gp.GaussianProcessCCA`
    used to require: each embedding $Z_i$ is exactly linear in $X_i$
    throughout fitting, so it inherits :class:`~cca_zoo._base.BaseModel`'s
    plain ``transform``/``weights`` machinery unmodified, and ``weights``
    genuinely are the sparse canonical weight vectors — not a placeholder
    that raises ``NotImplementedError`` the way it does for
    :class:`~cca_zoo.tree.TreeCCA`.

    Note:
        Like every EY-loss model, $\mathcal{L}_{EY}$ is not convex in $W$
        jointly (only each single coordinate's restriction is, in the
        limited sense of being an exactly-solvable quartic), so coordinate
        descent is only guaranteed to reach a stationary point, and
        different ``random_state`` initialisations can land on different
        ones — the same caveat that already applies to
        :class:`~cca_zoo.linear.gradient.CCAEY`'s gradient descent.

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means. Default is True.
        alpha: Overall elastic-net penalty strength. Default is 1.0.
        l1_ratio: Elastic-net mixing parameter in ``[0, 1]``; 0 is pure
            ridge, 1 is pure lasso. Default is 0.5.
        max_iter: Maximum number of full coordinate-descent sweeps (every
            view, feature, and component once each). Default is 100.
        tol: Convergence tolerance on the penalised objective's change
            between consecutive sweeps. Default is 1e-6.
        random_state: Seed for the initial weights.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((200, 20))
        >>> X2 = rng.standard_normal((200, 15))
        >>> model = ElasticNetCCA(latent_dimensions=2, alpha=0.1).fit([X1, X2])
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

    def fit(self, views: list[ArrayLike], y: None = None) -> ElasticNetCCA:
        """Fit ElasticNetCCA by cyclic coordinate descent on the EY loss.

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
        weights, _ = coordinate_descent_ey(
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
