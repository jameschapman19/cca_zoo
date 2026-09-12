r"""Shared machinery for the Eckart-Young (EY) unconstrained CCA objective.

This is the loss used by :class:`~cca_zoo.linear.gradient.CCAEY`,
:class:`~cca_zoo.linear.gradient.MCCAEY`, :class:`~cca_zoo.deep.DCCAEY`, and
:class:`~cca_zoo.tree.TreeCCA`: an unconstrained (no manifold projection
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

import numpy as np


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


def ey_diag_hessian(
    Z_i: np.ndarray,
    V: np.ndarray,
    n_views: int,
    n_minus_1: int,
    floor_percentile: float,
) -> np.ndarray:
    r"""Diagonal (per-sample) approximation of the EY loss's Hessian.

    The exact Hessian of $\mathcal{L}_{EY}$ w.r.t. one view's embedding
    $Z_i$ (holding the other views fixed) is

    $$
    \frac{\partial^2 \mathcal{L}_{EY}}{\partial Z_i^2}
        = \frac{4}{M(n-1)}(V-1)\,P + \frac{8}{M^2(n-1)^2}\, Z_i Z_i^\top
    $$

    where $P = I - \tfrac1n\mathbf{1}\mathbf{1}^\top$ is the centring
    projection (needed because the EY loss re-centres every embedding
    internally, so it is invariant to shifting $Z_i$ by a constant — the
    true Hessian must annihilate that direction) and $V$ is the (diagonal
    of the) mean auto-covariance. This is an $(n, n)$ matrix — using its
    diagonal as a per-sample weight is the same simplification every
    P-IRLS-based GAM/GLM solver already makes (treating observations as
    independent); the ``P`` term drops out of the diagonal (its diagonal
    entries are all $1 - 1/n$, folded into the same additive constant as
    the $(V-1)$ term).

    That diagonal is usable directly only after floor-damping it: near the
    loss's own well-conditioned fixed point ($V \approx 1$), the $(V-1)$
    term vanishes and the diagonal collapses to just $Z_{i,m}^2$ for each
    sample $m$ — the diagonal slice of a rank-1 matrix, an extremely poor
    per-sample curvature estimate for most samples (verified empirically:
    the per-sample ratio ``gradient / raw diagonal`` swings across several
    orders of magnitude with sign changes). Flooring at a high percentile
    of its own values — rather than a fixed constant, which would need
    re-tuning for every ``(n_samples, n_views)`` combination — is a
    self-calibrating Levenberg-Marquardt-style damping: it behaves like an
    (approximately) uniform weight for typical samples and only lets the
    Hessian's real signal through for high-leverage outliers.

    Used by both :class:`~cca_zoo.gam.GAMCCA` (as a ``Ridge``/``RidgeCV``
    ``sample_weight``) and :class:`~cca_zoo.gp.GaussianProcessCCA` (as a
    ``GaussianProcessRegressor`` per-sample ``alpha``, the heteroscedastic-
    noise parameter that plays the same "how much to trust this
    observation" role there).

    Args:
        Z_i: Current embedding for this view, shape (n_samples, k).
        V: Current (k, k) mean auto-covariance matrix (see
            :func:`ey_cross_covariance`).
        n_views: Number of views, $M$.
        n_minus_1: $n - 1$, the sample-covariance denominator.
        floor_percentile: Percentile (0-100) of the raw diagonal used as
            the damping floor.

    Returns:
        Array of shape (n_samples, k): positive per-sample weights, one
        per latent component.
    """
    v_diag = np.diag(V)
    a_coef = 4.0 / (n_views * n_minus_1) * (v_diag - 1.0)
    b_coef = 8.0 / (n_views**2 * n_minus_1**2)
    raw = a_coef[None, :] + b_coef * Z_i**2
    floor = np.maximum(np.percentile(raw, floor_percentile, axis=0), 1e-10)
    result: np.ndarray = np.maximum(raw, floor)
    return result


def _solve_quartic_coordinate(c4: float, c3: float, c2: float, c1: float, lasso: float) -> float:
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
    :func:`coordinate_descent_ey`'s docstring for the derivation; verified
    against direct evaluation of :func:`ey_loss`).

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

    def _f(w: float) -> float:
        return c4 * w**4 + c3 * w**3 + c2 * w**2 + c1 * w + lasso * abs(w)

    return min(candidates, key=_f)


def coordinate_descent_ey(
    bases: list[np.ndarray],
    k: int,
    alpha: float,
    l1_ratio: float,
    max_iter: int,
    tol: float,
    rng: np.random.Generator,
    ridge_matrices: list[np.ndarray | None] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    r"""Fit per-view linear-in-basis coefficients directly minimising the EY loss.

    Finds $B_1, \dots, B_M$ (embeddings $Z_i = \text{bases}_i B_i$)
    minimising

    $$
    \mathcal{L}_{EY}(Z_1, \dots, Z_M)
        + \sum_i \left( \alpha \rho \|B_i\|_1
        + \tfrac{1}{2}\alpha(1-\rho) \sum_c B_i[:,c]^\top M_i B_i[:,c] \right)
    $$

    ($\rho$ = ``l1_ratio``, $M_i$ = ``ridge_matrices[i]``, identity if
    ``None``) by **cyclic coordinate descent** — the same algorithm
    :class:`~sklearn.linear_model.ElasticNet` itself uses for ordinary
    (squared-error) elastic net, updating one scalar coefficient at a time
    to its exact minimiser with every other coefficient held fixed.

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

    Because every update is the *exact* per-coordinate minimiser of the
    *exact* (not linearised or diagonal-Hessian-approximated) EY loss, this
    needs no post-hoc whitening/decorrelation step: each $Z_i$ stays exactly
    linear in a *fixed* basis throughout fitting (never a Newton-step
    working response fit with a black-box regressor). Used by
    :class:`~cca_zoo.sparse.ElasticNetCCA` (``bases`` = the raw centred
    views, ``ridge_matrices=None``), :class:`~cca_zoo.gam.GAMCCA`
    (``bases`` = each view's centred spline basis, ``ridge_matrices=None``),
    and :class:`~cca_zoo.gp.GaussianProcessCCA` (``bases`` = each view's
    centred cross-kernel against its inducing points, ``ridge_matrices`` =
    the centred inducing-point kernel matrix, giving the RKHS-norm
    $B_i[:,c]^\top M_i B_i[:,c]$ penalty a Gaussian process's own posterior
    mean actually minimises, rather than a plain $\|B_i\|_2^2$ that would
    ignore the kernel's geometry).

    Args:
        bases: Fixed per-view design matrices, each already column-centred
            so $Z_i = \text{bases}_i B_i$ is automatically zero-mean, one
            per view.
        k: Number of latent dimensions.
        alpha: Overall elastic-net penalty strength.
        l1_ratio: Elastic-net mixing parameter in ``[0, 1]``; 0 is pure ridge.
        max_iter: Maximum number of full coordinate-descent sweeps.
        tol: Convergence tolerance on the penalised objective's change
            between consecutive sweeps.
        rng: Random generator for the initial coefficients.
        ridge_matrices: Per-view coupling matrix for the ridge term, each
            ``(bases[i].shape[1], bases[i].shape[1])`` and symmetric PSD, or
            ``None`` for a plain (identity, i.e. $\|B_i\|_2^2$) ridge
            penalty on that view. ``None`` (the default) uses a plain ridge
            penalty for every view.

    Returns:
        Tuple ``(coefficients, representations)``: ``coefficients[i]`` has
        shape ``(bases[i].shape[1], k)``; ``representations[i] =
        bases[i] @ coefficients[i]``, shape ``(n_samples, k)``.
    """
    m = len(bases)
    n = bases[0].shape[0]
    n_minus_1 = n - 1
    a0 = 1.0 / (m * n_minus_1)
    lasso = alpha * l1_ratio
    ridge = alpha * (1.0 - l1_ratio)
    if ridge_matrices is None:
        ridge_matrices = [None] * m

    coefficients = cheap_orthonormal_projection_weights(bases, k, None, rng)
    representations = [b @ c for b, c in zip(bases, coefficients)]
    total = sum(representations)
    col_sq_norms = [np.sum(b**2, axis=0) for b in bases]
    diag_m = [
        np.diag(rm) if rm is not None else np.ones(basis.shape[1])
        for rm, basis in zip(ridge_matrices, bases)
    ]
    # mb[i] = ridge_matrices[i] @ coefficients[i]; for a plain (identity)
    # ridge penalty this always equals coefficients[i] itself, so no extra
    # state is kept for those views (mb[i] stays None; see the "cross_jc"
    # branch below).
    mb: list[np.ndarray | None] = [
        rm @ coef if rm is not None else None
        for rm, coef in zip(ridge_matrices, coefficients)
    ]

    prev_obj = np.inf
    for _ in range(max_iter):
        for i, (basis, coef, rm) in enumerate(
            zip(bases, coefficients, ridge_matrices)
        ):
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
                m_jj = diag_m[i][j]
                for c in range(k):
                    w0 = coef[j, c]
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

                    cross_jc = 0.0 if mb[i] is None else mb[i][j, c] - m_jj * w0

                    w_new = _solve_quartic_coordinate(
                        c4=p4,
                        c3=p3,
                        c2=p2 + q2 + 0.5 * ridge * m_jj,
                        c1=p1 + q1 + ridge * cross_jc,
                        lasso=lasso,
                    )

                    delta = w_new - w0
                    if delta != 0.0:
                        coef[j, c] = w_new
                        zi[:, c] += xj * delta
                        total[:, c] += xj * delta
                        if rm is not None:
                            mb[i][:, c] += rm[:, j] * delta

        penalty = 0.0
        for coef, rm in zip(coefficients, ridge_matrices):
            penalty += alpha * l1_ratio * np.sum(np.abs(coef))
            if rm is None:
                penalty += 0.5 * alpha * (1.0 - l1_ratio) * np.sum(coef**2)
            else:
                penalty += 0.5 * alpha * (1.0 - l1_ratio) * sum(
                    coef[:, c] @ rm @ coef[:, c] for c in range(k)
                )
        obj = ey_loss(representations)["objective"] + penalty
        if abs(prev_obj - obj) < tol:
            break
        prev_obj = obj

    return coefficients, representations
