r"""Shared machinery for the Eckart-Young (EY) unconstrained CCA objective.

This is the loss used by :class:`~cca_zoo.linear.gradient.CCAEY` (2 or more
views), :class:`~cca_zoo.linear.gradient.StochasticCCAEY`,
:class:`~cca_zoo.deep.DCCAEY`, and :class:`~cca_zoo.tree.TreeCCA`: an
unconstrained (no manifold projection
required) stand-in for canonical correlation analysis that is a stationary
point exactly at the canonical directions.

For $M$ views with (possibly non-orthonormal) embeddings
$Z_1, \dots, Z_M$, each $(n, k)$, define the mean pairwise
cross-covariance and mean auto-covariance:

$$
C = \frac{1}{M} \sum_{i, j} \operatorname{Cov}(Z_i, Z_j), \qquad
V = \frac{1}{M} \sum_i \operatorname{Cov}(Z_i, Z_i)
$$

(the sum for $C$ ranges over *all* ordered pairs, including
$i = j$). The EY loss to minimise is:

$$
\mathcal{L}_{EY} = -2 \operatorname{tr}(C) + \operatorname{tr}(V V)
$$

References:
    Chapman, J., Wells, L., & Lawry Aguila, A. (2024). Unconstrained
    Stochastic CCA: Unifying Multiview and Self-Supervised Learning.
    arXiv:2310.01012.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.linalg
from scipy.sparse.linalg import LinearOperator


def ey_cross_covariance(
    representations: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    r"""Mean pairwise cross-covariance and mean auto-covariance of M embeddings.

    Args:
        representations: List of M arrays, each of shape (n_samples, k).
            Need not be pre-centred; centring is performed internally.

    Returns:
        Tuple ``(C, V)``, each of shape (k, k):
        ``C`` is the mean of $\operatorname{Cov}(Z_i, Z_j)$ over all
        ordered pairs (including $i = j$); ``V`` is the mean of
        $\operatorname{Cov}(Z_i, Z_i)$.
    """
    n = representations[0].shape[0]
    m = len(representations)
    centred = [z - z.mean(axis=0) for z in representations]
    k = centred[0].shape[1]
    C = np.zeros((k, k))
    V = np.zeros((k, k))
    for zi in centred:
        V += zi.T @ zi / (n - 1)
        for zj in centred:
            C += zi.T @ zj / (n - 1)
    return C / m, V / m


def ey_loss(representations: list[np.ndarray]) -> dict[str, float]:
    """Compute the EY loss and its reward/penalty components.

    Args:
        representations: List of M arrays, each of shape (n_samples, k).

    Returns:
        Dictionary with ``"objective"`` (``-rewards + penalties``, to be
        minimised), ``"rewards"`` (``2 * tr(C)``), and ``"penalties"``
        (``tr(V @ V)``).
    """
    C, V = ey_cross_covariance(representations)
    rewards = float(np.trace(2.0 * C))
    penalties = float(np.trace(V @ V))
    return {
        "objective": -rewards + penalties,
        "rewards": rewards,
        "penalties": penalties,
    }


def weight_gram_mean(weights: list[np.ndarray]) -> np.ndarray:
    r"""Mean weight Gram matrix $B = \frac{1}{M} \sum_i W_i^\top W_i$.

    Args:
        weights: Per-view weight matrices, each of shape (p_i, k).

    Returns:
        Matrix of shape (k, k).
    """
    total: np.ndarray = sum(w.T @ w for w in weights) / len(weights)
    return total


def random_orthonormal_weights(
    views: list[np.ndarray], latent_dimensions: int, rng: np.random.Generator
) -> list[np.ndarray]:
    r"""Cheap, data-independent initial weights with orthonormal columns.

    Each view's weight matrix is the $Q$ factor of a QR decomposition of
    an i.i.d. standard normal matrix, so $W_i^\top W_i = I$ exactly, before
    any gradient step and without looking at the data at all. This matches
    the structure of :class:`~cca_zoo.linear.gradient.PLSEY`'s own penalty,
    which drives weights towards orthonormality directly in weight space
    (see :func:`weight_gram_mean`) — the loss's own fixed point is already
    the natural initial point's shape.

    Args:
        views: Per-view arrays; only ``.shape[1]`` (feature count) is used.
        latent_dimensions: Requested number of latent dimensions.
        rng: Random generator.

    Returns:
        List of weight matrices, each $(p_i, k)$ with orthonormal columns,
        where $k = \min(\text{latent\_dimensions}, p_i)$.
    """
    weights = []
    for v in views:
        p = v.shape[1]
        k = min(latent_dimensions, p)
        w, _ = np.linalg.qr(rng.standard_normal((p, k)))
        weights.append(w)
    return weights


def cheap_orthonormal_projection_weights(
    views: list[np.ndarray],
    latent_dimensions: int,
    batch_size: int | None,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    r"""Cheap initial weights giving unit-variance projections on one batch.

    Classical CCA whitens each view with a full $(p, p)$ eigendecomposition
    of its covariance before fitting; that full-batch pass is exactly what
    the EY reformulation exists to avoid (see
    :class:`~cca_zoo.linear.gradient.CCAEY`). This is a cheap substitute
    usable only at initialisation: draw random directions, project one
    mini-batch, and QR-orthonormalise the resulting $(n, k)$ projection
    instead of the $(p, p)$ data covariance, then pull that
    orthonormalisation back into the weight matrix via the QR's $(k, k)$
    triangular factor — one small QR and one $k \times k$ solve per view,
    independent of $p$. Concretely, for random directions $W_0$,
    mini-batch $X$, and $X W_0 = QR$:

    $$
    W = W_0 R^{-1}, \qquad X W = X W_0 R^{-1} = Q R R^{-1} = Q
    $$

    so the resulting projections are exactly orthonormal ($Q^\top Q = I$)
    on that mini-batch — a cheap stand-in for the reward term's ideal
    starting point ($V \approx I$; see :func:`ey_cross_covariance`) and
    the natural match for :class:`~cca_zoo.linear.gradient.CCAEY`'s own
    fixed point. This orthonormalises only the *first* mini-batch, though:
    every later step draws an independent fresh batch, so this does not,
    by itself, prevent the ``c=0`` divergence risk noted in that class's
    docstring (empirically confirmed: it does not measurably postpone it
    either) — ``c`` or ``batch_size`` remain the actual remedy for that.

    Args:
        views: Per-view arrays, each $(n, p_i)$.
        latent_dimensions: Requested number of latent dimensions.
        batch_size: Mini-batch size used for the initial projection.
            ``None`` uses the full dataset.
        rng: Random generator.

    Returns:
        List of weight matrices, each $(p_i, k)$.
    """
    n = views[0].shape[0]
    bs = n if batch_size is None else min(batch_size, n)
    idx = rng.choice(n, bs, replace=False)
    weights = []
    for v in views:
        p = v.shape[1]
        k = min(latent_dimensions, p)
        w0, _ = np.linalg.qr(rng.standard_normal((p, k)))
        z0 = v[idx] @ w0
        _, r = np.linalg.qr(z0)
        weights.append(w0 @ np.linalg.solve(r, np.eye(k)))
    return weights


def random_orthogonal_embedding(
    Xc: np.ndarray, k: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Unit-variance random-orthogonal initial embedding and its projection.

    Draws ``k`` random orthogonal directions in feature space (independent of
    the data's principal directions) and rescales them so each initial
    component has unit variance. The unit-variance scaling is what matters
    for a well-conditioned, non-vanishing EY gradient from round zero;
    orthogonality keeps the initial cross-component covariance at zero. Used
    as the fixed starting point for nonlinear encoders trained by functional
    gradient boosting (:class:`~cca_zoo.tree.TreeCCA`,
    :class:`~cca_zoo.gam.GAMCCA`), which — unlike a linear map — have no
    natural "zero" to start from.

    Args:
        Xc: Mean-centred training view, shape (n_samples, n_features).
        k: Number of components. Must not exceed ``n_features``.
        rng: Random generator used to draw the orthogonal directions.

    Returns:
        Tuple ``(base_margin, projection)``: ``base_margin`` has shape
        (n_samples, k) and is the initial embedding for training;
        ``projection`` has shape (n_features, k) and reproduces the same
        unit-variance embedding for unseen data via ``Xc_new @ projection``.
    """
    n, p = Xc.shape
    W, _ = np.linalg.qr(rng.standard_normal((p, k)))
    Z = Xc @ W
    scale = np.linalg.norm(Z, axis=0, keepdims=True) / np.sqrt(n - 1)
    projection = (W / scale).astype(np.float32)
    base_margin = (Z / scale).astype(np.float32)
    return base_margin, projection


def rescale_grads_to_target_std(
    grads: list[np.ndarray], target_std: float = 0.1
) -> list[np.ndarray]:
    r"""Rescale a set of per-view EY gradients to a common target standard deviation.

    The analytic EY gradient (:func:`ey_grad_z`) has magnitude $O(1/n)$
    (from its ``4 / (M (n - 1))`` prefactor), far smaller than the natural
    scale of a boosted-tree leaf value. Used unscaled as a functional
    gradient-boosting target, a single round would then contribute a
    negligible increment relative to the encoder's starting embedding, no
    matter the learning rate. Rescaling by one shared scalar restores a
    well-conditioned target for :class:`~cca_zoo.tree.TreeCCA`'s trees.
    Since the same scalar is applied to every view, this changes only the
    effective step size, not the gradient's direction or relative
    cross-view magnitudes.

    Args:
        grads: One gradient array per view, each (n_samples, k).
        target_std: Target standard deviation. Default is 0.1.

    Returns:
        List of rescaled gradients, same dtype as the input.
    """
    scale = max(max(float(g.std()) for g in grads), 1e-6)
    return [g / scale * target_std for g in grads]


def ey_grad_z(representations: list[np.ndarray]) -> list[np.ndarray]:
    r"""Gradient of the EY loss w.r.t. each embedding (M-view generalised).

    $$
    \frac{\partial \mathcal{L}_{EY}}{\partial Z_k}
        = \frac{4}{M (n - 1)} \left( \tilde{Z}_k V - S \right)
    $$

    where $\tilde{Z}_k$ is the centred k-th embedding,
    $S = \sum_i \tilde{Z}_i$, and $V$ is the mean auto-covariance
    (see :func:`ey_cross_covariance`). Verified against finite-difference
    gradients of :func:`ey_loss` for M = 2, 3, 4.

    Args:
        representations: List of M arrays, each of shape (n_samples, k).

    Returns:
        List of M gradient arrays, each of shape (n_samples, k), one per view.
    """
    n = representations[0].shape[0]
    m = len(representations)
    centred = [z - z.mean(axis=0) for z in representations]
    total = sum(centred)
    _, V = ey_cross_covariance(representations)
    scale = 4.0 / (m * (n - 1))
    return [scale * (zc @ V - total) for zc in centred]


def _solve_quartic_coordinate(
    c4: float, c3: float, c2: float, c1: float, lasso: float, positive: bool = False
) -> float:
    r"""Exact global minimiser of one elastic-net-penalised coordinate update.

    Minimises $F(w) = c_4 w^4 + c_3 w^3 + c_2 w^2 + c_1 w + \lambda |w|$ over
    the scalar $w$ (the ridge penalty's contribution is already folded into
    $c_2$/$c_1$ by the caller — see :func:`coordinate_descent_ey`). Unlike
    ordinary least-squares coordinate descent (e.g. sklearn's
    ``ElasticNet``, where the restriction of the squared-error loss to one
    coordinate is quadratic, giving the familiar closed-form soft-threshold
    update), the EY loss's penalty term $\operatorname{tr}(VV)$ is quadratic
    in $V$, which is itself quadratic in the coordinate being updated — so
    the restriction is exactly *quartic*, not quadratic (see
    :func:`coordinate_descent_ey`'s docstring for the derivation).

    $F \to +\infty$ as $w \to \pm\infty$ (``c4`` is a perfect square, so
    non-negative), so a finite global minimiser always exists. It is found
    exactly — with no local-optimum risk, whether or not $F$ is convex — by
    comparing $F$ at every stationary point of each smooth branch (the real
    roots of the cubic derivative on ``w > 0`` and on ``w < 0`` separately,
    since $|w|$'s derivative flips sign there) plus the $|w|$ kink at 0.

    Args:
        c4: Coefficient of the smooth quartic term.
        c3: Coefficient of the smooth cubic term.
        c2: Coefficient of the smooth quadratic term.
        c1: Coefficient of the smooth linear term.
        lasso: L1 penalty coefficient ($\ge 0$).
        positive: If True, restrict the search to $w \ge 0$ (sklearn's
            ``positive=True`` constraint on ``Lasso``/``ElasticNet``) by
            dropping the $w < 0$ branch entirely — $|w|$ on that branch is
            just $w$, so no other change is needed.

    Returns:
        The scalar $w$ exactly minimising $F$ (subject to $w \ge 0$ if
        ``positive``).
    """
    candidates = [0.0]
    branches = ((1.0, lasso),) if positive else ((1.0, lasso), (-1.0, -lasso))
    for sign, l1 in branches:
        roots = np.roots([4 * c4, 3 * c3, 2 * c2, c1 + l1])
        for r in roots:
            if abs(r.imag) < 1e-8 and sign * r.real > 0:
                candidates.append(float(r.real))

    def _f(w: float) -> float:
        return c4 * w**4 + c3 * w**3 + c2 * w**2 + c1 * w + lasso * abs(w)

    return min(candidates, key=_f)


def _ey_coordinate_smooth_quartic(
    xj: np.ndarray,
    a: float,
    a0: float,
    zi: np.ndarray,
    total: np.ndarray,
    v_other: np.ndarray,
    coef_row: np.ndarray,
    c: int,
    k: int,
) -> tuple[float, float, float, float]:
    r"""Quartic coefficients of the *unpenalised* EY loss restricted to one coordinate.

    Shared derivation used by both :func:`coordinate_descent_ey` (which adds
    the ridge penalty to ``c2`` and hands the result to
    :func:`_solve_quartic_coordinate` for an exact scalar solve) and
    :func:`group_coordinate_descent_ey` (which instead evaluates the
    quartic's derivative at the current point to get one entry of a row's
    gradient — see that function's docstring for why a whole-row update
    can't reuse the same closed-form scalar solve). Isolated here purely so
    the two call sites can't silently drift apart.

    Args:
        xj: The basis column for this coordinate's feature, shape (n,).
        a: $\|x_j\|^2$, precomputed by the caller.
        a0: The shared prefactor $1 / (M (n-1))$.
        zi: This view's current embedding, shape (n, k) — read-only here.
        total: $\sum_i Z_i$, shape (n, k) — read-only here.
        v_other: Mean auto-covariance contribution from every *other* view
            plus this view's residual-so-far, shape (k, k).
        coef_row: This feature's current coefficient row across components,
            shape (k,) — only ``coef_row[c]`` (the coordinate's own current
            value) is used, to reconstruct the residual excluding it.
        c: Index of the component (column) being restricted to.
        k: Number of latent dimensions.

    Returns:
        Tuple ``(p4, p3, smooth_c2, smooth_c1)``: the quartic, cubic,
        quadratic and linear coefficients of the unpenalised restriction
        $c_4 w^4 + c_3 w^3 + \text{smooth\_c2} \, w^2 + \text{smooth\_c1} \, w$.
    """
    w0 = coef_row[c]
    r_c = zi[:, c] - xj * w0
    s0_c = total[:, c] - xj * w0

    u_c = xj @ r_c
    v0_cc = v_other[c, c] + (r_c @ r_c) * a0
    x_s0c = xj @ s0_c

    other_c = [cc for cc in range(k) if cc != c]
    u_other = [xj @ zi[:, cc] for cc in other_c]
    v1_other = [v_other[c, cc] + (r_c @ zi[:, cc]) * a0 for cc in other_c]

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

    return p4, p3, p2 + q2, p1 + q1


def coordinate_descent_ey(
    bases: list[np.ndarray],
    k: int,
    alpha: list[float],
    l1_ratio: list[float],
    max_iter: int,
    tol: float,
    rng: np.random.Generator,
    positive: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    r"""Fit per-view linear-in-basis coefficients directly minimising the EY loss.

    Finds $B_1, \dots, B_M$ (embeddings $Z_i = \text{bases}_i B_i$)
    minimising

    $$
    \mathcal{L}_{EY}(Z_1, \dots, Z_M)
        + \sum_i \left( \alpha \rho \|B_i\|_1
        + \tfrac{1}{2}\alpha(1-\rho) \|B_i\|_2^2 \right)
    $$

    ($\rho$ = ``l1_ratio``) by **cyclic coordinate descent** — the same
    algorithm :class:`~sklearn.linear_model.ElasticNet` itself uses for
    ordinary (squared-error) elastic net, updating one scalar coefficient at
    a time to its exact minimiser with every other coefficient held fixed.

    This is a genuine departure from ``ElasticNet``'s own coordinate
    descent, not a re-use of it: for ordinary least squares, the loss
    restricted to a single coordinate is a plain quadratic, giving the
    familiar closed-form soft-threshold update. Restricting $\mathcal{L}_{EY}$
    to a single coordinate $w = B_i[j, c]$ instead gives an **exact
    quartic**: the penalty term $\operatorname{tr}(VV)$ is quadratic in the
    auto-covariance $V$, which is itself quadratic in $w$ through
    $V[c,c] = \dots + w^2\|{\text{bases}_i[:,j]}\|^2/(M(n-1)) + \dots$, so
    squaring it produces a $w^4$ term (the reward term
    $-2\operatorname{tr}(C)$ and the cross terms $V[c,c']^2$, $c'\neq c$,
    stay quadratic in $w$; only the "self" term $V[c,c]^2$ contributes the
    quartic and cubic pieces). Each coordinate's exact minimiser is found by
    :func:`_solve_quartic_coordinate`.

    Every update is the *exact* per-coordinate minimiser of the *exact*
    (not linearised) EY loss, so $Z_i$ stays exactly linear in a *fixed*
    basis throughout fitting. Used by :class:`~cca_zoo.sparse.ElasticNetCCA`
    (``bases`` = the raw centred views), the only model needing an L1
    (lasso) penalty; a purely ridge-penalised fixed-basis fit is instead
    solved by :class:`~cca_zoo.gam.GAMCCA`'s joint trust-region Newton-CG
    solve or :class:`~cca_zoo.gp.GaussianProcessCCA`'s L-BFGS-B.

    Args:
        bases: Fixed per-view design matrices, each already column-centred
            so $Z_i = \text{bases}_i B_i$ is automatically zero-mean, one
            per view.
        k: Number of latent dimensions.
        alpha: Overall elastic-net penalty strength, one per view.
        l1_ratio: Elastic-net mixing parameter in ``[0, 1]`` per view; 0 is
            pure ridge.
        max_iter: Maximum number of full coordinate-descent sweeps.
        tol: Convergence tolerance on the penalised objective's change
            between consecutive sweeps.
        rng: Random generator for the initial coefficients.
        positive: If True, constrain every coefficient to be non-negative
            (sklearn's ``Lasso``/``ElasticNet`` ``positive=True``) — see
            :func:`_solve_quartic_coordinate`.

    Returns:
        Tuple ``(coefficients, representations)``: ``coefficients[i]`` has
        shape ``(bases[i].shape[1], k)``; ``representations[i] =
        bases[i] @ coefficients[i]``, shape ``(n_samples, k)``.
    """
    m = len(bases)
    n = bases[0].shape[0]
    n_minus_1 = n - 1
    a0 = 1.0 / (m * n_minus_1)
    lasso = [a * r for a, r in zip(alpha, l1_ratio)]
    ridge = [a * (1.0 - r) for a, r in zip(alpha, l1_ratio)]

    coefficients = cheap_orthonormal_projection_weights(bases, k, None, rng)
    representations = [b @ c for b, c in zip(bases, coefficients)]
    total = sum(representations)
    col_sq_norms = [np.sum(b**2, axis=0) for b in bases]

    prev_obj = np.inf
    for _ in range(max_iter):
        for i, (basis, coef) in enumerate(zip(bases, coefficients)):
            zi = representations[i]
            v_other = (
                sum(
                    representations[a].T @ representations[a]
                    for a in range(m)
                    if a != i
                )
                * a0
            )
            for j in range(basis.shape[1]):
                a = col_sq_norms[i][j]
                if a < 1e-12:
                    continue
                xj = basis[:, j]
                for c in range(k):
                    w0 = coef[j, c]
                    p4, p3, smooth_c2, smooth_c1 = _ey_coordinate_smooth_quartic(
                        xj, a, a0, zi, total, v_other, coef[j, :], c, k
                    )

                    w_new = _solve_quartic_coordinate(
                        c4=p4,
                        c3=p3,
                        c2=smooth_c2 + 0.5 * ridge[i],
                        c1=smooth_c1,
                        lasso=lasso[i],
                        positive=positive,
                    )

                    delta = w_new - w0
                    if delta != 0.0:
                        coef[j, c] = w_new
                        zi[:, c] += xj * delta
                        total[:, c] += xj * delta

        penalty = sum(
            alpha[i] * l1_ratio[i] * np.sum(np.abs(c)) + 0.5 * ridge[i] * np.sum(c**2)
            for i, c in enumerate(coefficients)
        )
        obj = ey_loss(representations)["objective"] + penalty
        if abs(prev_obj - obj) < tol:
            break
        prev_obj = obj

    return coefficients, representations


def _group_penalty(
    coefficients: list[np.ndarray], alpha: list[float], l1_ratio: list[float]
) -> float:
    r"""Row-group elastic-net penalty on a list of per-view coefficient matrices.

    $$
    \sum_i \left( \alpha_i \rho_i \|B_i\|_{2,1}
        + \tfrac{1}{2} \alpha_i (1-\rho_i) \|B_i\|_F^2 \right)
    $$

    $\|B_i\|_{2,1} = \sum_j \|B_i[j, :]\|_2$ is the sum, over features, of
    each feature's coefficient-row Euclidean norm — the row-group analogue
    of $\|B_i\|_1$'s per-scalar absolute value, used by
    :func:`group_coordinate_descent_ey`.
    """
    return float(
        sum(
            a * r * np.sum(np.linalg.norm(c, axis=1))
            + 0.5 * a * (1.0 - r) * np.sum(c**2)
            for c, a, r in zip(coefficients, alpha, l1_ratio)
        )
    )


def _group_prox(u: np.ndarray, lasso: float, denom: float) -> np.ndarray:
    r"""Proximal operator of $\lambda \|\cdot\|_2$ at $u$, scaled by ``denom``.

    Exact minimiser of $\tfrac{\text{denom}}{2} \|w - u\|_2^2 + \lambda \|w\|_2$
    over the vector $w$: the classic group-lasso block soft-threshold,
    shrinking $u$'s *length* towards zero while keeping its direction.

    Args:
        u: Point to shrink, shape (k,).
        lasso: L1-analogue (group) penalty coefficient ($\ge 0$).
        denom: The quadratic term's coefficient ($> 0$).

    Returns:
        The shrunk vector, shape (k,) — exactly ``0`` once
        ``lasso >= denom * ||u||``.
    """
    norm_u = float(np.linalg.norm(u))
    if norm_u < 1e-15:
        return np.zeros_like(u)
    shrink = max(0.0, 1.0 - lasso / (denom * norm_u))
    return shrink * u


def group_coordinate_descent_ey(
    bases: list[np.ndarray],
    k: int,
    alpha: list[float],
    l1_ratio: list[float],
    max_iter: int,
    tol: float,
    rng: np.random.Generator,
    max_backtrack: int = 40,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    r"""Fit per-view linear-in-basis coefficients with a row-group elastic-net penalty.

    Same setting as :func:`coordinate_descent_ey` — embeddings
    $Z_i = \text{bases}_i B_i$ minimising the EY loss plus a penalty on
    $B_i$ — but with sklearn's :class:`~sklearn.linear_model.MultiTaskLasso`
    / :class:`~sklearn.linear_model.MultiTaskElasticNet` penalty in place of
    plain elastic net:

    $$
    \mathcal{L}_{EY}(Z_1, \dots, Z_M) + \sum_i \left(
        \alpha \rho \|B_i\|_{2,1} + \tfrac{1}{2}\alpha(1-\rho) \|B_i\|_F^2
    \right)
    $$

    where $\|B_i\|_{2,1} = \sum_j \|B_i[j,:]\|_2$ sums each *feature's*
    coefficient-row norm over all $k$ components (see :func:`_group_penalty`).
    Unlike the per-scalar lasso in :func:`coordinate_descent_ey`, this
    penalty is zero only when an entire row is zero, so a feature is either
    active in every latent dimension or in none — the natural sparsity
    pattern once a model has more than one component, instead of a feature
    surviving in component 1 but dropping out of component 2 for no
    principled reason.

    This is *not* a re-use of :func:`coordinate_descent_ey`'s exact
    per-scalar quartic solve, and can't be: sklearn's own multi-task solver
    gets a closed-form block update only because ordinary least squares is
    quadratic and *separable* across tasks (columns) for a fixed row — no
    cross terms between components. The EY loss's $\operatorname{tr}(VV)$
    penalty has no such luck: $V[c, c']$ for $c \neq c'$ is bilinear in a
    row's two entries $w_c, w_{c'}$, so a whole row's restriction is a
    genuinely coupled multivariate quartic with no closed-form joint
    minimiser for general $k$. Instead each row is updated by one step of
    **proximal gradient (ISTA) with backtracking line search**: the row's
    exact gradient at the current point is read off from
    :func:`_ey_coordinate_smooth_quartic`'s linear coefficient (the same
    quantity :func:`coordinate_descent_ey` evaluates the quartic at, here
    evaluated *only* at the current point rather than solved exactly), a
    trial step is taken by the row-group analogue of soft-thresholding
    (:func:`_group_prox`) at a quadratic majoriser with curvature ``L``,
    and ``L`` is doubled until the *exact* penalised objective (evaluated
    directly, not the majoriser) does not increase — a standard ISTA
    guarantee that terminates in $O(\log(1/\epsilon))$ doublings since
    ``L -> infinity`` collapses the step to zero. Every accepted row update
    is therefore a genuine decrease of the true objective, giving the same
    monotonic-descent guarantee as :func:`coordinate_descent_ey`, just
    without that function's additional guarantee of exactness within a step.

    Args:
        bases: Fixed per-view design matrices, each already column-centred,
            one per view.
        k: Number of latent dimensions.
        alpha: Overall penalty strength, one per view.
        l1_ratio: Mixing parameter in ``[0, 1]`` per view; 0 is pure (Frobenius)
            ridge, 1 is pure row-group lasso.
        max_iter: Maximum number of full coordinate-descent sweeps.
        tol: Convergence tolerance on the penalised objective's change
            between consecutive sweeps.
        rng: Random generator for the initial coefficients.
        max_backtrack: Maximum line-search doublings of ``L`` per row
            before giving up and leaving that row at its previous value for
            this sweep (mathematically this only happens at
            float-precision-level step sizes, never as a genuine failure to
            improve).

    Returns:
        Tuple ``(coefficients, representations)``, same shapes as
        :func:`coordinate_descent_ey`.
    """
    m = len(bases)
    n = bases[0].shape[0]
    a0 = 1.0 / (m * (n - 1))
    lasso = [a * r for a, r in zip(alpha, l1_ratio)]
    ridge = [a * (1.0 - r) for a, r in zip(alpha, l1_ratio)]

    coefficients = cheap_orthonormal_projection_weights(bases, k, None, rng)
    representations = [b @ c for b, c in zip(bases, coefficients)]
    total = sum(representations)
    col_sq_norms = [np.sum(b**2, axis=0) for b in bases]

    cur_obj = ey_loss(representations)["objective"] + _group_penalty(
        coefficients, alpha, l1_ratio
    )
    prev_obj = np.inf
    for _ in range(max_iter):
        for i, (basis, coef) in enumerate(zip(bases, coefficients)):
            zi = representations[i]
            v_other = (
                sum(
                    representations[a].T @ representations[a]
                    for a in range(m)
                    if a != i
                )
                * a0
            )
            for j in range(basis.shape[1]):
                a = col_sq_norms[i][j]
                if a < 1e-12:
                    continue
                xj = basis[:, j]
                w0_row = coef[j, :].copy()

                grads = np.empty(k)
                for c in range(k):
                    p4, p3, smooth_c2, smooth_c1 = _ey_coordinate_smooth_quartic(
                        xj, a, a0, zi, total, v_other, w0_row, c, k
                    )
                    w0c = w0_row[c]
                    grads[c] = (
                        4 * p4 * w0c**3
                        + 3 * p3 * w0c**2
                        + 2 * smooth_c2 * w0c
                        + smooth_c1
                    )

                lipschitz = max(a0 * a, 1e-6)
                for _try in range(max_backtrack):
                    denom = lipschitz + ridge[i]
                    u = (lipschitz * w0_row - grads) / denom
                    w_new_row = _group_prox(u, lasso[i], denom)
                    delta = w_new_row - w0_row
                    if np.any(delta != 0.0):
                        for c in range(k):
                            zi[:, c] += xj * delta[c]
                            total[:, c] += xj * delta[c]
                        coef[j, :] = w_new_row

                    trial_obj = ey_loss(representations)["objective"] + _group_penalty(
                        coefficients, alpha, l1_ratio
                    )
                    if trial_obj <= cur_obj + 1e-12:
                        cur_obj = trial_obj
                        break

                    if np.any(delta != 0.0):
                        for c in range(k):
                            zi[:, c] -= xj * delta[c]
                            total[:, c] -= xj * delta[c]
                        coef[j, :] = w0_row
                    lipschitz *= 2.0

        if abs(prev_obj - cur_obj) < tol:
            break
        prev_obj = cur_obj

    return coefficients, representations


def omp_coordinate_descent_ey(
    bases: list[np.ndarray],
    k: int,
    n_nonzero_coefs: list[int],
    max_iter: int,
    tol: float,
    rng: np.random.Generator,
    refit_sweeps: int = 20,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    r"""Fit per-view linear-in-basis coefficients by greedy forward selection (EY loss).

    The EY-loss analogue of sklearn's
    :class:`~sklearn.linear_model.OrthogonalMatchingPursuit`: instead of a
    continuous penalty trading off sparsity against fit
    (:func:`coordinate_descent_ey`, :func:`group_coordinate_descent_ey`),
    each view's active feature set is grown one feature at a time up to a
    fixed budget ``n_nonzero_coefs[i]``, and the active coefficients are
    re-solved to their exact joint optimum after every addition — the same
    two-phase structure (`select`, then `least-squares refit on the active
    set`) as classical OMP, with "least squares" replaced by the EY loss's
    own exact per-coordinate quartic solve
    (:func:`_ey_coordinate_smooth_quartic` /
    :func:`_solve_quartic_coordinate`, ``lasso=0``).

    Feature selection reuses OMP's own criterion: classical OMP picks the
    column most correlated with the current residual, i.e. the column
    whose inclusion gives the largest-magnitude gradient of the squared-error
    loss at zero. Here that is the EY loss's own
    per-coordinate linear coefficient (:func:`_ey_coordinate_smooth_quartic`'s
    ``smooth_c1`` return value *is* $\partial\mathcal{L}_{EY}/\partial w$ at
    $w=0$, since the quartic's higher-order terms vanish there), generalised
    from a scalar to a $k$-vector (one entry per latent dimension) and
    ranked by Euclidean norm.

    One EY-specific wrinkle with no OLS counterpart: the EY loss's
    zero-embedding point is degenerate — $Z_i \equiv 0$ for *every* view
    simultaneously is already a stationary point (see
    :func:`ey_grad_z`), so growing every view's support from a literal
    empty start would leave the very first choice, for the very first
    view, with nothing to score candidates against. This is resolved by
    warm-starting *every* view with a small dense fit
    (:func:`cheap_orthonormal_projection_weights`) before any view's
    support is touched, then regrowing each view's support from scratch,
    one view at a time, against the *other* views' (still meaningfully
    nonzero) current embeddings — after which every view has been visited
    at least once, so later rounds bootstrap off genuinely sparse fits
    rather than the initial dense one.

    Args:
        bases: Fixed per-view design matrices, each already column-centred,
            one per view.
        k: Number of latent dimensions.
        n_nonzero_coefs: Target number of active features for each view
            (already resolved/validated by the caller — see
            :class:`~cca_zoo.sparse.OrthogonalMatchingPursuitCCA`).
        max_iter: Maximum number of outer rounds cycling through every view
            and regrowing its support from scratch.
        tol: Convergence tolerance on the (unpenalised) EY objective's
            change between consecutive outer rounds, and between
            consecutive refit sweeps after each single feature addition.
        rng: Random generator for the initial dense warm start.
        refit_sweeps: Maximum coordinate-descent sweeps used to re-solve
            the active coefficients after each feature addition.

    Returns:
        Tuple ``(coefficients, representations)``, same shapes as
        :func:`coordinate_descent_ey`. Every row outside a view's active
        set is exactly zero.
    """
    m = len(bases)
    n = bases[0].shape[0]
    a0 = 1.0 / (m * (n - 1))
    n_features = [b.shape[1] for b in bases]
    col_sq_norms = [np.sum(b**2, axis=0) for b in bases]

    coefficients = cheap_orthonormal_projection_weights(bases, k, None, rng)
    representations = [b @ c for b, c in zip(bases, coefficients)]
    total = sum(representations)

    prev_obj = np.inf
    for _ in range(max_iter):
        for i, basis in enumerate(bases):
            target = min(n_nonzero_coefs[i], n_features[i])
            zi = representations[i]
            total -= zi
            zi = np.zeros_like(zi)
            coef = np.zeros_like(coefficients[i])
            representations[i] = zi
            coefficients[i] = coef

            v_other = (
                sum(
                    representations[a].T @ representations[a]
                    for a in range(m)
                    if a != i
                )
                * a0
            )

            active: list[int] = []
            inactive = [j for j in range(n_features[i]) if col_sq_norms[i][j] >= 1e-12]
            zero_row = np.zeros(k)
            for _step in range(target):
                if not inactive:
                    break
                best_j, best_score = inactive[0], -1.0
                for j in inactive:
                    a = col_sq_norms[i][j]
                    xj = basis[:, j]
                    score = float(
                        np.linalg.norm(
                            [
                                _ey_coordinate_smooth_quartic(
                                    xj, a, a0, zi, total, v_other, zero_row, c, k
                                )[3]
                                for c in range(k)
                            ]
                        )
                    )
                    if score > best_score:
                        best_score, best_j = score, j
                active.append(best_j)
                inactive.remove(best_j)

                refit_prev = np.inf
                for _ in range(refit_sweeps):
                    for j in active:
                        a = col_sq_norms[i][j]
                        xj = basis[:, j]
                        for c in range(k):
                            w0 = coef[j, c]
                            p4, p3, smooth_c2, smooth_c1 = (
                                _ey_coordinate_smooth_quartic(
                                    xj, a, a0, zi, total, v_other, coef[j, :], c, k
                                )
                            )
                            w_new = _solve_quartic_coordinate(
                                c4=p4, c3=p3, c2=smooth_c2, c1=smooth_c1, lasso=0.0
                            )
                            delta = w_new - w0
                            if delta != 0.0:
                                coef[j, c] = w_new
                                zi[:, c] += xj * delta
                                total[:, c] += xj * delta
                    refit_obj = ey_loss(representations)["objective"]
                    if abs(refit_prev - refit_obj) < tol:
                        break
                    refit_prev = refit_obj

        obj = ey_loss(representations)["objective"]
        if abs(prev_obj - obj) < tol:
            break
        prev_obj = obj

    return coefficients, representations


def _penalty_matrix(penalty: float | np.ndarray, size: int) -> np.ndarray:
    """A view's penalty as a matrix: a scalar ridge, per-column ridges, or a matrix."""
    penalty = np.asarray(penalty, dtype=float)
    if penalty.ndim == 2:
        return penalty
    return np.diag(np.broadcast_to(penalty, size))


def penalised_gram_ey_gep(
    gram: np.ndarray, view: np.ndarray, penalties: Sequence[float | np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    r"""The generalized eigenproblem whose solution minimises the penalised EY loss.

    For fixed column-centred bases $\Phi_i$ and coefficients stacked as
    $W = [B_1; \dots; B_M]$, the quadratically penalised EY loss is

    $$
    -2\operatorname{tr}(W^\top A W) + \operatorname{tr}\big((W^\top B W)^2\big)
        + \tfrac12 \operatorname{tr}(W^\top R W),
    $$

    with $A = \Phi^\top\Phi / (M(n-1))$ over the concatenated bases (every
    cross- and auto-covariance block — ``gram``), $B$ its block diagonal
    (each view's own auto-covariance), and $R = \operatorname{blockdiag}(R_i)$
    each view's penalty — a ridge $\lambda_i I$ (:class:`~cca_zoo.gam.MARSCCA`)
    or a P-spline difference penalty (:class:`~cca_zoo.gam.GAMCCA`). Its
    stationarity condition $(A - R/4)\,W = B W (W^\top B W)$ is solved by
    the generalized eigenvectors $(A - R/4)\,U = B U \operatorname{diag}(\mu)$,
    $U^\top B U = I$, scaled as $W = U\operatorname{diag}(\sqrt{\mu})$, where
    the loss equals $-\sum \mu^2$: the global minimum over $k$ components
    takes the $k$ largest positive eigenvalues (a component whose eigenvalue
    is not positive is zero). The fixed-basis penalised EY fit is therefore
    a closed-form eigenproblem, the same one regularised MCCA solves.

    Args:
        gram: ``A``, shape (D, D), positive definite on each view's block
            (see :func:`full_rank_reparametrisation` otherwise).
        view: View index of each of the D stacked columns.
        penalties: One per view: a scalar ridge, one ridge per column, or a
            symmetric positive semi-definite matrix.

    Returns:
        ``(A - R/4, B)``, the two sides of the eigenproblem.
    """
    b = np.where(view[:, None] == view[None, :], gram, 0.0)
    sizes = np.bincount(view)
    penalty = scipy.linalg.block_diag(
        *[_penalty_matrix(p, int(size)) for p, size in zip(penalties, sizes)]
    )
    return gram - penalty / 4, b


def penalised_basis_ey_gep(
    bases: list[np.ndarray], penalties: Sequence[float | np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:func:`penalised_gram_ey_gep` from column-centred bases.

    Returns:
        ``(A - R/4, B, view)``, with the view index of each stacked column.
    """
    gram, view = _stacked_gram(bases)
    return (*penalised_gram_ey_gep(gram, view, penalties), view)


def _stacked_gram(bases: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """``Phi' Phi / (M (n - 1))`` over the concatenated bases, and column views."""
    stacked = np.hstack(bases)
    view = np.repeat(np.arange(len(bases)), [basis.shape[1] for basis in bases])
    gram: np.ndarray = stacked.T @ stacked / (len(bases) * (stacked.shape[0] - 1))
    return gram, view


def _jacobi_scale(bases: list[np.ndarray]) -> np.ndarray:
    """Per-column scale giving every stacked basis column unit norm."""
    scale: np.ndarray = 1.0 / np.linalg.norm(np.hstack(bases), axis=0)
    return scale


def _jacobi_scaled(
    bases: list[np.ndarray], penalties: Sequence[float | np.ndarray]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Bases with unit-norm columns, and the penalties that keep the problem unchanged.

    Rescaling columns by ``s`` rescales their coefficients by ``1/s``, so the
    same fit needs the penalty ``diag(s) R diag(s)`` (the eigenproblem is
    invariant; only its conditioning improves, which the Cholesky
    factorisation of ``B`` needs when column norms span orders of
    magnitude).
    """
    scale = _jacobi_scale(bases)
    split = np.cumsum([b.shape[1] for b in bases])[:-1]
    per_view = np.split(scale, split)
    return (
        [b * sv for b, sv in zip(bases, per_view)],
        [
            _penalty_matrix(p, len(sv)) * np.outer(sv, sv)
            for p, sv in zip(penalties, per_view)
        ],
    )


def _jacobi(
    lhs: np.ndarray, rhs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(lhs, rhs)`` scaled by ``1/sqrt(diag(rhs))`` on both sides, and the scale.

    The eigenvalues are unchanged; the Cholesky factorisation of ``rhs`` is
    conditioned when its diagonal spans orders of magnitude.
    """
    scale = 1.0 / np.sqrt(np.diag(rhs))
    outer = np.outer(scale, scale)
    return lhs * outer, rhs * outer, scale


def _top_eigenpairs(
    lhs: np.ndarray, rhs: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Top ``k`` eigenpairs of ``(lhs, rhs)``, largest first."""
    size = lhs.shape[0]
    lhs_s, rhs_s, scale = _jacobi(lhs, rhs)
    mu, u = scipy.linalg.eigh(
        lhs_s, rhs_s, subset_by_index=(max(size - k, 0), size - 1)
    )
    return mu[::-1], u[:, ::-1] * scale[:, None]


def _top_eigenvalues(lhs: np.ndarray, rhs: np.ndarray, k: int) -> np.ndarray:
    """Top ``k`` eigenvalues of ``(lhs, rhs)``, largest first."""
    size = lhs.shape[0]
    lhs_s, rhs_s, _ = _jacobi(lhs, rhs)
    mu = scipy.linalg.eigh(
        lhs_s,
        rhs_s,
        eigvals_only=True,
        subset_by_index=(max(size - k, 0), size - 1),
    )
    result: np.ndarray = mu[::-1]
    return result


def penalised_gram_ey_closed_form(
    gram: np.ndarray, view: np.ndarray, k: int, penalties: Sequence[float | np.ndarray]
) -> list[np.ndarray]:
    """Globally optimal penalised-EY coefficients from the stacked Gram matrix.

    Solves :func:`penalised_gram_ey_gep` for its ``k`` largest eigenvalues.

    Args:
        gram: Stacked ``Phi' Phi / (M (n - 1))``, positive definite per view.
        view: View index of each stacked column.
        k: Number of latent dimensions.
        penalties: One per view, as :func:`penalised_gram_ey_gep` takes.

    Returns:
        Per-view coefficients, each of shape ``(d_i, k)``; components beyond
        the number of positive eigenvalues are zero.
    """
    lhs, rhs = penalised_gram_ey_gep(gram, view, penalties)
    mu, u = _top_eigenpairs(lhs, rhs, k)
    w = np.zeros((lhs.shape[0], k))
    w[:, : len(mu)] = u * np.sqrt(np.maximum(mu, 0.0))
    return [w[view == i] for i in range(int(view.max()) + 1)]


def penalised_basis_ey_closed_form(
    bases: list[np.ndarray], k: int, penalties: Sequence[float | np.ndarray]
) -> list[np.ndarray]:
    """Globally optimal penalised-EY coefficients on fixed bases.

    Args:
        bases: Column-centred per-view design matrices of full column rank.
        k: Number of latent dimensions.
        penalties: One per view, as :func:`penalised_gram_ey_gep` takes.

    Returns:
        Per-view coefficients, ``coefficients[i]`` of shape
        ``(bases[i].shape[1], k)``; components beyond the number of positive
        eigenvalues are zero.
    """
    return penalised_gram_ey_closed_form(*_stacked_gram(bases), k, penalties)


def penalised_basis_ey_min_loss(
    bases: list[np.ndarray], k: int, penalties: Sequence[float | np.ndarray]
) -> float:
    """Minimum penalised-EY loss on fixed bases: ``-sum(mu**2)`` over the top k.

    Args:
        bases: Column-centred per-view design matrices of full column rank.
        k: Number of latent dimensions.
        penalties: One per view, as :func:`penalised_gram_ey_gep` takes.

    Returns:
        The loss :func:`penalised_basis_ey_closed_form`'s coefficients
        attain, from the eigenvalues alone.
    """
    gram, view = _stacked_gram(bases)
    mu = _top_eigenvalues(*penalised_gram_ey_gep(gram, view, penalties), k)
    return -float(np.sum(np.maximum(mu, 0.0) ** 2))


# Gram eigenvalues below this fraction of the largest count as null in
# full_rank_reparametrisation: forming the Gram leaves rounding of order
# d * eps * the largest eigenvalue (~1e-13 for d ~ 1000), and a direction
# whose data signal is under ~1e-5 of the strongest (the square root of this)
# is negligible to the embedding while numerically unreliable to fit.
_GRAM_RANK_TOL = 1e-10

# Row-space directions whose Gram eigenvalue is below this fraction of the
# largest are projected out of the null basis once more, against the basis
# itself (see full_rank_reparametrisation); above it eigh's contamination is
# already below eps / 1e-4 ~ 2e-12.
_GRAM_REFINE_BELOW = 1e-4


def full_rank_reparametrisation(
    basis: np.ndarray | LinearOperator,
    gram: np.ndarray,
    penalty_factor: np.ndarray,
    penalty_norm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Rewrite a rank-deficient penalised basis as an equivalent full-rank one.

    With the basis's Gram matrix $G = \Phi^\top\Phi = V \Lambda V^\top$ and
    penalty $R = F^\top F$, split coefficients as $w = T a + N c$ with $T$
    spanning $\Phi$'s row space (eigenvalues above
    :data:`_GRAM_RANK_TOL` of the largest) and $N$ its null space. The
    embedding $\Phi w = \Phi T a$ depends on $a$ alone, so for any $a$ the
    best $c$ minimises only the penalty, the least-squares problem
    $\min_c \lVert F (T a + N c) \rVert$; its minimum-norm solution
    $c^\star = -(F N)^{+} F T a$ leaves $w = L a$ with
    $L = T - N (F N)^{+} F T$ and penalty $(F L)^\top (F L)$ — an equivalent
    full-rank problem whose Gram is $T^\top G T$ (as $\Phi N = 0$ and
    $T^\top L = I$; taking it from $L^\top G L$ instead would reintroduce $G$'s
    null-space block, zero in exact arithmetic but with rounding-sized
    negative eigenvalues that $L$'s correction amplifies into a negative
    diagonal). It is what lets a
    P-spline basis keep splines with no data under them (the difference
    penalty fills them in) and keep every feature's partition of unity
    (which centring makes collinear), the identifiability ``mgcv`` resolves
    by reparametrisation. The split comes from the $d \times d$ Gram,
    never an SVD of the $n \times d$ basis.

    Three numerical choices keep $L$ accurate. The Gram squares the basis's
    condition number, so eigh leaves each null vector contaminated along a
    row-space direction $t_j$ by about $\epsilon \lambda_{\max} / \lambda_j$
    — up to ~1e-6 at the rank cutoff, which the correction below amplifies
    into large cancelling coefficients. One projection step against the basis
    itself, $N \leftarrow N - T_w \Lambda_w^{-1} (\Phi T_w)^\top (\Phi N)$
    over the weak directions $T_w$ (eigenvalues below
    :data:`_GRAM_REFINE_BELOW` of the largest), removes it to the accuracy
    of an SVD of $\Phi$ at the cost of a product with $|T_w| + |N|$ columns.
    The least-squares problem is The least-squares problem is
    solved on the penalty's factor $F$ rather than by pseudo-inverting
    $N^\top R N$, whose condition number is the square of $F N$'s. And its
    singular values count as zero below $\sqrt{\epsilon}\,\lVert F \rVert$,
    measured against the penalty's own scale rather than $F N$'s: a
    direction null in both $\Phi$ and $R$ — every feature's constant, for a
    P-spline — has a rounding-sized singular value there rather than zero,
    and a cutoff relative to $F N$ itself inverted it into coefficients of
    1e13–1e15, harmless to the embedding but ruinous to each feature's own
    term.

    Args:
        basis: The basis $\Phi$, shape (n, d), as anything supporting
            ``basis @ matrix`` (e.g. a sparse
            :class:`~scipy.sparse.linalg.LinearOperator`).
        gram: Its Gram matrix ``basis.T @ basis``, shape (d, d).
        penalty_factor: ``F`` with penalty ``R = F.T @ F``, shape (q, d).
        penalty_norm: ``F``'s spectral norm.

    Returns:
        ``(lift, row, reduced_penalty)``: shapes (d, r), (d, r) and (r, r).
        The reduced problem's Gram is ``row.T @ gram @ row`` (across views,
        ``row_i.T @ gram_ij @ row_j``), and its coefficients ``a`` map back as
        ``lift @ a``.
    """
    eigenvalues, vectors = np.linalg.eigh(gram)
    row_mask = eigenvalues > eigenvalues[-1] * _GRAM_RANK_TOL
    row, null = vectors[:, row_mask], vectors[:, ~row_mask]
    weak = row_mask & (eigenvalues < eigenvalues[-1] * _GRAM_REFINE_BELOW)
    if null.shape[1] and weak.any():
        weak_row = vectors[:, weak]
        overlap = (basis @ weak_row).T @ (basis @ null)
        null = np.linalg.qr(null - weak_row @ (overlap / eigenvalues[weak, None]))[0]
    fu, fs, fvt = np.linalg.svd(penalty_factor @ null, full_matrices=False)
    keep = fs > np.sqrt(np.finfo(float).eps) * penalty_norm
    correction = fvt[keep].T @ (
        (fu[:, keep].T @ (penalty_factor @ row)) / fs[keep, None]
    )
    lift = row - null @ correction
    factor_lift = penalty_factor @ lift
    return lift, row, factor_lift.T @ factor_lift
