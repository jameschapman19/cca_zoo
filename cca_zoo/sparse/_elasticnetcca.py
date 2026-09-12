r"""ElasticNetCCA — sparse linear CCA via coordinate descent directly on the EY loss."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils._param_validation import Interval

from cca_zoo._base import BaseModel
from cca_zoo._utils._ey import cheap_orthonormal_projection_weights, ey_loss


def _solve_quartic_coordinate(c4: float, c3: float, c2: float, c1: float, lasso: float) -> float:
    r"""Exact global minimiser of one elastic-net-penalised coordinate update.

    Minimises $F(w) = c_4 w^4 + c_3 w^3 + c_2 w^2 + c_1 w + \lambda |w|$ over
    the scalar $w$ (the ridge penalty's contribution is already folded into
    $c_2$ by the caller). Unlike ordinary least-squares coordinate descent
    (e.g. sklearn's ``ElasticNet``, where the restriction of the squared-error
    loss to one coordinate is quadratic, giving the familiar closed-form
    soft-threshold update), the EY loss's penalty term
    $\operatorname{tr}(VV)$ is quadratic in $V$, which is itself quadratic in
    the coordinate being updated — so the restriction is exactly *quartic*,
    not quadratic (see the module docstring for the derivation; verified
    against direct evaluation of :func:`~cca_zoo._utils._ey.ey_loss`).

    $F \to +\infty$ as $w \to \pm\infty$ (``c4`` is a perfect square, so
    non-negative), so a finite global minimiser always exists. It is found
    exactly — with no local-optimum risk, whether or not $F$ is convex — by
    comparing $F$ at every stationary point of each smooth branch (the real
    roots of the cubic derivative on ``w > 0`` and on ``w < 0`` separately,
    since $|w|$'s derivative flips sign there) plus the $|w|$ kink at 0.

    Args:
        c4, c3, c2: Coefficients of the smooth quartic/cubic/quadratic terms.
        c1: Coefficient of the smooth linear term.
        lasso: L1 penalty coefficient ($\ge 0$).

    Returns:
        The scalar $w$ exactly minimising $F$.
    """
    candidates = [0.0]
    for sign, l1 in ((1.0, lasso), (-1.0, -lasso)):
        roots = np.roots([4 * c4, 3 * c3, 2 * c2, c1 + l1])
        for r in roots:
            if abs(r.imag) < 1e-8 and sign * r.real > 0:
                candidates.append(float(r.real))

    def _F(w: float) -> float:
        return c4 * w**4 + c3 * w**3 + c2 * w**2 + c1 * w + lasso * abs(w)

    return min(candidates, key=_F)


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
    and $\rho$ is ``l1_ratio``. Unlike :class:`~cca_zoo.linear.gradient.CCAEY`,
    which reaches its optimum by mini-batch gradient descent, this is fit by
    **cyclic coordinate descent** — the same algorithm
    :class:`~sklearn.linear_model.ElasticNet` itself uses — updating one
    scalar weight at a time to its exact minimiser with every other weight
    held fixed, and repeating until the penalised objective stops moving.

    This is a genuine departure from ``ElasticNet``'s own coordinate descent,
    not a re-use of it: for ordinary (squared-error) elastic net, the loss
    restricted to a single coordinate is a plain quadratic, so each update
    has the familiar closed-form soft-threshold solution. Here, restricting
    $\mathcal{L}_{EY}$ to a single coordinate $w = W_i[j, c]$ (every other
    weight fixed) gives an **exact quartic**, not a quadratic: the penalty
    term $\operatorname{tr}(VV)$ is quadratic in the auto-covariance $V$,
    which is itself quadratic in $w$ through
    $V[c,c] = \dots + w^2\|x_j\|^2 / (M(n-1)) + \dots$, so squaring it
    produces a $w^4$ term. (The reward term $-2\operatorname{tr}(C)$ and the
    cross terms $V[c,c']^2$, $c'\neq c$, stay quadratic in $w$; only the
    "self" term $V[c,c]^2$ contributes the quartic and cubic pieces.) Each
    coordinate's exact global minimiser — smooth quartic-plus-ridge part,
    kinked by the L1 term at $w=0$ — is found via
    :func:`_solve_quartic_coordinate`: the real roots of a cubic (the smooth
    part's derivative, one branch for $w>0$, one for $w<0$) compared against
    the $w=0$ kink, with no line search or step size and no approximation.

    Because every update is the *exact* per-coordinate minimiser of the
    *exact* (not linearised or diagonal-Hessian-approximated) EY loss, this
    needs no post-hoc whitening/decorrelation step of the kind
    :class:`~cca_zoo.gam.GAMCCA` and :class:`~cca_zoo.gp.GaussianProcessCCA`
    require to compensate for their diagonal-Hessian approximation (see
    those classes' docstrings): each embedding $Z_i$ is exactly linear in
    $X_i$ throughout fitting (never a Newton-step working response fit
    with a black-box regressor), so it inherits :class:`~cca_zoo._base.BaseModel`'s
    plain ``transform``/``weights`` machinery unmodified, and ``weights``
    genuinely are the sparse canonical weight vectors — not a placeholder
    that raises ``NotImplementedError`` the way it does for
    :class:`~cca_zoo.tree.TreeCCA` or :class:`~cca_zoo.gam.GAMCCA`.

    Note:
        Each coordinate update recomputes its exact quartic coefficients
        from the current embeddings directly (an $O(Mnk^2)$ pass per view,
        per sweep, to refresh the cross-view auto-covariance the update
        needs — see ``fit``), rather than maintaining those sufficient
        statistics incrementally the way ``ElasticNet``'s own solver does
        for its (much cheaper, quadratic) coordinate updates. This trades
        some speed for a direct, easily-checked implementation; it does not
        affect correctness or the fixed point reached.

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
        k = self.latent_dimensions
        m = len(views_)
        n = views_[0].shape[0]
        n_minus_1 = n - 1
        a0 = 1.0 / (m * n_minus_1)
        lasso = self.alpha * self.l1_ratio
        ridge = self.alpha * (1.0 - self.l1_ratio)

        rng = np.random.default_rng(self.random_state)
        weights = cheap_orthonormal_projection_weights(views_, k, None, rng)
        representations = [v @ w for v, w in zip(views_, weights)]
        total = sum(representations)
        col_sq_norms = [np.sum(v**2, axis=0) for v in views_]

        prev_obj = np.inf
        for _ in range(self.max_iter):
            for i, (Xi, Wi) in enumerate(zip(views_, weights)):
                zi = representations[i]
                v_other = (
                    sum(
                        representations[a].T @ representations[a]
                        for a in range(m)
                        if a != i
                    )
                    * a0
                )
                for j in range(Xi.shape[1]):
                    a = col_sq_norms[i][j]
                    if a < 1e-12:
                        continue
                    xj = Xi[:, j]
                    for c in range(k):
                        w0 = Wi[j, c]
                        r_c = zi[:, c] - xj * w0
                        s0_c = total[:, c] - xj * w0

                        u_c = xj @ r_c
                        v0_cc = v_other[c, c] + (r_c @ r_c) * a0
                        x_s0c = xj @ s0_c

                        other_c = [cc for cc in range(k) if cc != c]
                        u_other = [xj @ zi[:, cc] for cc in other_c]
                        v1_other = [
                            v_other[c, cc] + (r_c @ zi[:, cc]) * a0 for cc in other_c
                        ]

                        p4 = (a0 * a) ** 2
                        p3 = 4 * a0**2 * a * u_c
                        p2 = (
                            4 * a0**2 * u_c**2
                            + 2 * a0 * a * v0_cc
                            + 2 * a0**2 * sum(uo**2 for uo in u_other)
                        )
                        p1 = 4 * a0 * u_c * v0_cc + 4 * a0 * sum(
                            uo * v1 for uo, v1 in zip(u_other, v1_other)
                        )
                        q2 = -2 * a0 * a
                        q1 = -4 * a0 * x_s0c

                        w_new = _solve_quartic_coordinate(
                            c4=p4,
                            c3=p3,
                            c2=p2 + q2 + 0.5 * ridge,
                            c1=p1 + q1,
                            lasso=lasso,
                        )

                        delta = w_new - w0
                        if delta != 0.0:
                            Wi[j, c] = w_new
                            zi[:, c] += xj * delta
                            total[:, c] += xj * delta

            penalty = sum(
                self.alpha * self.l1_ratio * np.sum(np.abs(w))
                + 0.5 * self.alpha * (1.0 - self.l1_ratio) * np.sum(w**2)
                for w in weights
            )
            obj = ey_loss(representations)["objective"] + penalty
            if abs(prev_obj - obj) < self.tol:
                break
            prev_obj = obj

        self.weights_: list[np.ndarray] = weights
        return self
