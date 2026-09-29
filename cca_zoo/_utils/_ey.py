r"""The Eckart-Young (EY) CCA objective and its solvers.

For $M$ views with embeddings $Z_1, \dots, Z_M$, each $(n, k)$, let

$$
C = \frac{1}{M} \sum_{i, j} \operatorname{Cov}(Z_i, Z_j), \qquad
V = \frac{1}{M} \sum_i \operatorname{Cov}(Z_i, Z_i),
$$

with $C$ summed over all ordered pairs including $i = j$. The EY loss

$$
\mathcal{L}_{EY} = -2 \operatorname{tr}(C) + \operatorname{tr}(V V)
$$

is unconstrained and stationary exactly at the canonical directions.

References:
    Chapman, J., Wells, L., & Lawry Aguila, A. (2024). Unconstrained
    Stochastic CCA: Unifying Multiview and Self-Supervised Learning.
    arXiv:2310.01012.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import scipy.linalg
from scipy.sparse.linalg import LinearOperator


def ey_cross_covariance(
    representations: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Mean pairwise cross-covariance ``C`` and mean auto-covariance ``V``.

    Args:
        representations: One array of shape (n_samples, k) per view; centred
            internally.

    Returns:
        ``(C, V)``, each of shape (k, k).
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
    """The EY loss and its terms.

    Args:
        representations: One array of shape (n_samples, k) per view.

    Returns:
        ``{"objective": -rewards + penalties, "rewards": 2 tr(C),
        "penalties": tr(V V)}``.
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
    """Mean weight Gram matrix ``sum_i W_i' W_i / M``, shape (k, k)."""
    total: np.ndarray = sum(w.T @ w for w in weights) / len(weights)
    return total


def random_orthonormal_weights(
    views: list[np.ndarray], n_components: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Random weights with orthonormal columns, one matrix per view.

    Each is the Q factor of a Gaussian matrix, of shape
    ``(n_features_i, min(n_components, n_features_i))``.
    """
    weights = []
    for v in views:
        p = v.shape[1]
        k = min(n_components, p)
        w, _ = np.linalg.qr(rng.standard_normal((p, k)))
        weights.append(w)
    return weights


def cheap_orthonormal_projection_weights(
    views: list[np.ndarray],
    n_components: int,
    batch_size: int | None,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Random weights whose projections of one batch are orthonormal.

    With random directions $W_0$ and $X W_0 = QR$ on a batch $X$,
    $W = W_0 R^{-1}$ gives $X W = Q$: a cheap stand-in for whitening that
    costs one small QR per view.

    Args:
        views: Per-view arrays, each of shape (n_samples, n_features_i).
        n_components: Number of latent dimensions.
        batch_size: Rows used for the projection; ``None`` uses all.
        rng: Random generator.

    Returns:
        One weight matrix of shape (n_features_i, k) per view.
    """
    n = views[0].shape[0]
    bs = n if batch_size is None else min(batch_size, n)
    idx = rng.choice(n, bs, replace=False)
    weights = []
    for v in views:
        p = v.shape[1]
        k = min(n_components, p)
        w0, _ = np.linalg.qr(rng.standard_normal((p, k)))
        z0 = v[idx] @ w0
        _, r = np.linalg.qr(z0)
        weights.append(w0 @ np.linalg.solve(r, np.eye(k)))
    return weights


def random_orthogonal_embedding(
    Xc: np.ndarray, k: int, rng: np.random.Generator, std: float = 1.0
) -> np.ndarray:
    """Random orthogonal embedding of a view with a given standard deviation.

    Breaks the symmetry for the tree encoders' first round: the EY gradient
    is zero at an all-zero embedding.

    Args:
        Xc: Centred view, shape (n_samples, n_features).
        k: Number of components, at most ``n_features``.
        rng: Random generator.
        std: Standard deviation of each component. Default is 1.

    Returns:
        The embedding, shape (n_samples, k), as ``float32``.
    """
    n, p = Xc.shape
    W, _ = np.linalg.qr(rng.standard_normal((p, k)))
    Z = Xc @ W
    scale = np.linalg.norm(Z, axis=0, keepdims=True) / np.sqrt(n - 1) / std
    embedding: np.ndarray = (Z / scale).astype(np.float32)
    return embedding


def ey_grad_z(representations: list[np.ndarray]) -> list[np.ndarray]:
    r"""Gradient of the EY loss with respect to each embedding.

    $$
    \frac{\partial \mathcal{L}_{EY}}{\partial Z_i}
        = \frac{4}{M (n - 1)} \left( \tilde{Z}_i V - S \right),
    $$

    with $\tilde{Z}_i$ the centred embedding and $S = \sum_j \tilde{Z}_j$.

    Args:
        representations: One array of shape (n_samples, k) per view.

    Returns:
        One gradient of shape (n_samples, k) per view.
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
    """Global minimiser of ``c4 w^4 + c3 w^3 + c2 w^2 + c1 w + lasso |w|``.

    Restricted to one coordinate the EY loss is quartic, not quadratic as in
    least squares. The minimiser is the best of the real stationary points
    on each side of zero and the kink at zero; ``c4 >= 0`` guarantees one
    exists. ``positive`` restricts the search to ``w >= 0``.
    """
    candidates = [0.0]
    branches = ((1.0, lasso),) if positive else ((1.0, lasso), (-1.0, -lasso))
    for sign, l1 in branches:
        roots = np.roots([4 * c4, 3 * c3, 2 * c2, c1 + l1])
        candidates.extend(
            float(r.real) for r in roots if abs(r.imag) < 1e-8 and sign * r.real > 0
        )

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
    """Quartic coefficients ``(c4, c3, c2, c1)`` of the EY loss along one coordinate.

    Shared by :func:`coordinate_descent_ey` and
    :func:`group_coordinate_descent_ey` so the derivation lives in one place.
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


# Fractions of alpha at which the penalised solvers fit in turn.
_PENALTY_PATH = (0.0, 0.1, 0.3, 1.0)


def _along_penalty_path(
    sweeps: Callable[[list[float]], tuple[list[np.ndarray], int, bool]],
    coefficients: list[np.ndarray],
    alpha: list[float],
    penalty: Callable[[], float],
) -> tuple[int, bool]:
    """Run ``sweeps`` at each fraction of ``alpha`` in turn, warm-started.

    ``sweeps`` updates ``coefficients`` in place and returns the embeddings,
    the sweeps it ran and whether it converged. All-zero weights have
    objective zero, so a fit ending above that is replaced by them.

    Returns:
        ``(n_iter, converged)`` of the fit at the full penalty.
    """
    for scale in _PENALTY_PATH:
        representations, n_iter, converged = sweeps([a * scale for a in alpha])
    if ey_loss(representations)["objective"] + penalty() > 0.0:
        for c in coefficients:
            c[:] = 0.0
    return n_iter, converged


def coordinate_descent_ey(
    bases: list[np.ndarray],
    k: int,
    alpha: list[float],
    l1_ratio: list[float],
    max_iter: int,
    tol: float,
    rng: np.random.Generator,
    positive: bool = False,
) -> tuple[list[np.ndarray], int, bool]:
    r"""Elastic-net penalised EY fit on fixed bases by exact coordinate descent.

    Minimises, over $Z_i = \text{bases}_i B_i$,

    $$
    \mathcal{L}_{EY} + \sum_i \left( \alpha_i \rho_i \|B_i\|_1
        + \tfrac{1}{2}\alpha_i(1-\rho_i) \|B_i\|_2^2 \right),
    $$

    updating one coefficient at a time to the exact minimiser of its quartic
    restriction (:func:`_solve_quartic_coordinate`). All-zero weights are a
    local minimum once every view reaches them, so the penalty is raised to
    ``alpha`` along a path from zero, each stage warm-started from the last,
    as glmnet does; a fit whose objective ends above zero's returns zeros.
    Used by :class:`~cca_zoo.sparse.ElasticNetCCA`.

    Args:
        bases: Column-centred design matrix of each view.
        k: Number of latent dimensions.
        alpha: Penalty strength of each view.
        l1_ratio: L1 share of each view's penalty, in ``[0, 1]``.
        max_iter: Maximum full sweeps.
        tol: Tolerance on the change in the objective between sweeps.
        rng: Random generator for the initial coefficients.
        positive: Constrain every coefficient to be non-negative.

    Returns:
        ``(coefficients, n_iter, converged)``: per-view coefficients of shape
        (n_basis_i, k), and the sweeps at the full penalty and whether they
        met ``tol`` before ``max_iter``.
    """
    coefficients = cheap_orthonormal_projection_weights(bases, k, None, rng)
    n_iter, converged = _along_penalty_path(
        lambda scaled: _elastic_net_sweeps(
            bases, coefficients, scaled, l1_ratio, max_iter, tol, positive
        ),
        coefficients,
        alpha,
        lambda: _elastic_net_penalty(coefficients, alpha, l1_ratio),
    )
    return coefficients, n_iter, converged


def _elastic_net_penalty(
    coefficients: list[np.ndarray], alpha: list[float], l1_ratio: list[float]
) -> float:
    """Elastic-net penalty summed over views."""
    return float(
        sum(
            a * r * np.sum(np.abs(c)) + 0.5 * a * (1.0 - r) * np.sum(c**2)
            for c, a, r in zip(coefficients, alpha, l1_ratio)
        )
    )


def _elastic_net_sweeps(
    bases: list[np.ndarray],
    coefficients: list[np.ndarray],
    alpha: list[float],
    l1_ratio: list[float],
    max_iter: int,
    tol: float,
    positive: bool,
) -> tuple[list[np.ndarray], int, bool]:
    """Exact coordinate sweeps of :func:`coordinate_descent_ey`, in place.

    Returns:
        The embedding of each view, shape (n_samples, k), the sweeps run and
        whether the objective settled within ``tol``.
    """
    m = len(bases)
    n = bases[0].shape[0]
    k = coefficients[0].shape[1]
    a0 = 1.0 / (m * (n - 1))
    lasso = [a * r for a, r in zip(alpha, l1_ratio)]
    ridge = [a * (1.0 - r) for a, r in zip(alpha, l1_ratio)]
    representations = [b @ c for b, c in zip(bases, coefficients)]
    total = sum(representations)
    col_sq_norms = [np.sum(b**2, axis=0) for b in bases]

    prev_obj = np.inf
    for n_iter in range(1, max_iter + 1):
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

        obj = ey_loss(representations)["objective"] + _elastic_net_penalty(
            coefficients, alpha, l1_ratio
        )
        if abs(prev_obj - obj) < tol:
            return representations, n_iter, True
        prev_obj = obj
    return representations, max_iter, False


def _group_penalty(
    coefficients: list[np.ndarray], alpha: list[float], l1_ratio: list[float]
) -> float:
    """Row-group elastic-net penalty summed over views."""
    return float(
        sum(
            a * r * np.sum(np.linalg.norm(c, axis=1))
            + 0.5 * a * (1.0 - r) * np.sum(c**2)
            for c, a, r in zip(coefficients, alpha, l1_ratio)
        )
    )


def _group_prox(u: np.ndarray, lasso: float, denom: float) -> np.ndarray:
    """Group soft-threshold: the minimiser of ``denom/2 ||w - u||^2 + lasso ||w||``."""
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
) -> tuple[list[np.ndarray], int, bool]:
    r"""Row-group elastic-net penalised EY fit on fixed bases.

    As :func:`coordinate_descent_ey` with sklearn's ``MultiTaskElasticNet``
    penalty, $\alpha\rho\|B_i\|_{2,1} + \tfrac12\alpha(1-\rho)\|B_i\|_F^2$,
    so each feature is active in every component or in none. A row's
    restriction is a coupled quartic with no closed-form minimiser, so each
    row takes a proximal-gradient step, backtracking until the EY loss is
    below its quadratic upper bound at the step. Accepting any step that
    lowers the penalised objective instead lets a row jump to zero past a
    nonzero minimum. The penalty follows the same path as
    :func:`coordinate_descent_ey`.

    Args:
        bases: Column-centred design matrix of each view.
        k: Number of latent dimensions.
        alpha: Penalty strength of each view.
        l1_ratio: Row-group share of each view's penalty, in ``[0, 1]``.
        max_iter: Maximum full sweeps.
        tol: Tolerance on the change in the objective between sweeps.
        rng: Random generator for the initial coefficients.
        max_backtrack: Maximum step halvings per row.

    Returns:
        ``(coefficients, n_iter, converged)`` as :func:`coordinate_descent_ey`.
    """
    coefficients = cheap_orthonormal_projection_weights(bases, k, None, rng)
    n_iter, converged = _along_penalty_path(
        lambda scaled: _group_sweeps(
            bases, coefficients, scaled, l1_ratio, max_iter, tol, max_backtrack
        ),
        coefficients,
        alpha,
        lambda: _group_penalty(coefficients, alpha, l1_ratio),
    )
    return coefficients, n_iter, converged


def _group_sweeps(
    bases: list[np.ndarray],
    coefficients: list[np.ndarray],
    alpha: list[float],
    l1_ratio: list[float],
    max_iter: int,
    tol: float,
    max_backtrack: int,
) -> tuple[list[np.ndarray], int, bool]:
    """Proximal row sweeps of :func:`group_coordinate_descent_ey`, in place.

    Returns:
        As :func:`_elastic_net_sweeps`.
    """
    m = len(bases)
    n = bases[0].shape[0]
    k = coefficients[0].shape[1]
    a0 = 1.0 / (m * (n - 1))
    lasso = [a * r for a, r in zip(alpha, l1_ratio)]
    ridge = [a * (1.0 - r) for a, r in zip(alpha, l1_ratio)]
    representations = [b @ c for b, c in zip(bases, coefficients)]
    total = sum(representations)
    col_sq_norms = [np.sum(b**2, axis=0) for b in bases]

    cur_loss = ey_loss(representations)["objective"]
    prev_obj = np.inf
    for n_iter in range(1, max_iter + 1):
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
                    step = np.outer(xj, delta)
                    zi += step
                    total += step
                    coef[j, :] = w_new_row

                    trial_loss = ey_loss(representations)["objective"]
                    bound = cur_loss + grads @ delta + 0.5 * lipschitz * (delta @ delta)
                    if trial_loss <= bound + 1e-12:
                        cur_loss = trial_loss
                        break

                    # Reject the step and backtrack with a larger curvature.
                    zi -= step
                    total -= step
                    coef[j, :] = w0_row
                    lipschitz *= 2.0

        cur_obj = cur_loss + _group_penalty(coefficients, alpha, l1_ratio)
        if abs(prev_obj - cur_obj) < tol:
            return representations, n_iter, True
        prev_obj = cur_obj
    return representations, max_iter, False


def omp_coordinate_descent_ey(
    bases: list[np.ndarray],
    k: int,
    n_nonzero_coefs: list[int],
    max_iter: int,
    tol: float,
    rng: np.random.Generator,
    refit_sweeps: int = 20,
) -> tuple[list[np.ndarray], int, bool]:
    """EY fit on fixed bases by greedy forward selection, as in OMP.

    Each view's active set grows one feature at a time up to its budget,
    choosing the feature with the largest EY gradient norm and refitting
    the active coefficients by exact coordinate descent after each addition.
    All views start from a dense warm start, since the all-zero embedding is
    a stationary point with nothing to select against. Rounds regrow every
    view's active set until the sets repeat.

    Args:
        bases: Column-centred design matrix of each view.
        k: Number of latent dimensions.
        n_nonzero_coefs: Active-set size of each view.
        max_iter: Maximum rounds of regrowing every view's active set.
        tol: Tolerance on the change in the EY loss between refit sweeps.
        rng: Random generator for the warm start.
        refit_sweeps: Maximum sweeps refitting the active set per addition.

    Returns:
        ``(coefficients, n_iter, converged)`` as :func:`coordinate_descent_ey`,
        counting rounds, converged when the active sets repeat; rows outside
        the active sets are exactly zero.
    """
    m = len(bases)
    n = bases[0].shape[0]
    a0 = 1.0 / (m * (n - 1))
    n_features = [b.shape[1] for b in bases]
    col_sq_norms = [np.sum(b**2, axis=0) for b in bases]

    coefficients = cheap_orthonormal_projection_weights(bases, k, None, rng)
    representations = [b @ c for b, c in zip(bases, coefficients)]
    total = sum(representations)

    prev_active_sets: list[list[int]] = []
    for n_iter in range(1, max_iter + 1):
        active_sets: list[list[int]] = []
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

            zero_row = np.zeros(k)

            def gradient_norm(j: int) -> float:
                """Norm of the EY gradient in feature j's (zero) coefficients."""
                return float(
                    np.linalg.norm(
                        [
                            _ey_coordinate_smooth_quartic(
                                basis[:, j],
                                col_sq_norms[i][j],
                                a0,
                                zi,
                                total,
                                v_other,
                                zero_row,
                                c,
                                k,
                            )[3]
                            for c in range(k)
                        ]
                    )
                )

            active: list[int] = []
            inactive = [j for j in range(n_features[i]) if col_sq_norms[i][j] >= 1e-12]
            for _step in range(min(target, len(inactive))):
                best_j = max(inactive, key=gradient_norm)
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
                            coef[j, c] = w_new
                            zi[:, c] += xj * (w_new - w0)
                            total[:, c] += xj * (w_new - w0)
                    refit_obj = ey_loss(representations)["objective"]
                    if abs(refit_prev - refit_obj) < tol:
                        break
                    refit_prev = refit_obj
            active_sets.append(sorted(active))

        if active_sets == prev_active_sets:
            return coefficients, n_iter, True
        prev_active_sets = active_sets
    return coefficients, max_iter, False


def _penalty_factor(penalty: float | np.ndarray, size: int) -> np.ndarray:
    """A view's penalty as its factor ``F``, with penalty ``F' F``.

    A scalar ridge or per-column ridges give ``diag(sqrt(ridge))``; a 2-D
    array is the factor itself. Penalties are carried as factors because
    squaring one squares its condition number, blurring the null space of a
    heavily weighted penalty.
    """
    penalty = np.asarray(penalty, dtype=float)
    if penalty.ndim == 2:
        return penalty
    return np.diag(np.sqrt(np.broadcast_to(penalty, size)))


def penalised_gram_ey_gep(
    gram: np.ndarray, view: np.ndarray, penalties: Sequence[float | np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    r"""The generalized eigenproblem solving the quadratically penalised EY fit.

    With coefficients stacked as $W$, the penalised EY loss on fixed bases is
    $-2\operatorname{tr}(W^\top A W) + \operatorname{tr}((W^\top B W)^2)
    + \tfrac12 \operatorname{tr}(W^\top R W)$, where $A$ is the stacked Gram
    matrix, $B$ its block diagonal and $R$ the block-diagonal penalty. Its
    minimiser is $W = U \operatorname{diag}(\sqrt{\mu})$ for the top positive
    generalized eigenpairs $(A - R/4) U = B U \operatorname{diag}(\mu)$,
    with loss $-\sum \mu^2$.

    Args:
        gram: ``A``, shape (D, D), positive definite on each view's block.
        view: View index of each stacked column.
        penalties: One per view: a scalar ridge, one ridge per column, or a
            factor ``F`` of shape (q, d_i) for the penalty ``F' F``.

    Returns:
        ``(A - R/4, B)``.
    """
    b = np.where(view[:, None] == view[None, :], gram, 0.0)
    sizes = np.bincount(view)
    penalty = scipy.linalg.block_diag(
        *[
            factor.T @ factor
            for factor in (
                _penalty_factor(p, int(size)) for p, size in zip(penalties, sizes)
            )
        ]
    )
    return gram - penalty / 4, b


def penalised_basis_ey_gep(
    bases: list[np.ndarray], penalties: Sequence[float | np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(lhs, rhs, view)``: :func:`penalised_gram_ey_gep` on centred bases."""
    gram, view = _stacked_gram(bases)
    return (*penalised_gram_ey_gep(gram, view, penalties), view)


def _stacked_gram(bases: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Stacked Gram ``Phi' Phi / (M (n - 1))`` of the bases, and each column's view."""
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
    """Bases with unit-norm columns and the correspondingly rescaled penalties.

    The eigenproblem is unchanged; only its conditioning improves.
    """
    scale = _jacobi_scale(bases)
    split = np.cumsum([b.shape[1] for b in bases])[:-1]
    per_view = np.split(scale, split)
    return (
        [b * sv for b, sv in zip(bases, per_view)],
        [_penalty_factor(p, len(sv)) * sv for p, sv in zip(penalties, per_view)],
    )


def _demmler_reinsch(
    gram: np.ndarray, view: np.ndarray, penalties: Sequence[float | np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""The penalised EY eigenproblem as a well-conditioned standard one.

    Each view's coordinates are whitened by its Gram block and rotated to
    diagonalise its penalty (the Demmler-Reinsch basis), so the problem
    becomes the symmetric eigenproblem of $T^\top A T - \operatorname{diag}(d)/4$
    with $w = T u$. A direction whose penalty eigenvalue $d$ exceeds the
    largest possible reward by $1/\sqrt{\epsilon}$ takes a weight below
    $\sqrt{\epsilon}$ of the others and is dropped: kept, its $d$ would set
    the eigensolver's absolute error, and a large smoothing parameter would
    swamp the eigenvalues that matter.

    Returns:
        ``(lhs, transform, view)``: the reduced symmetric matrix, the map from
        its coordinates to the stacked coefficients, and each kept
        coordinate's view.
    """
    n_views = int(view.max()) + 1
    # Whitened, the stacked Gram has unit diagonal blocks and norm at most
    # the number of views; no reward exceeds that.
    cutoff = 4 * n_views / np.sqrt(np.finfo(float).eps)
    blocks, kept_view = [], []
    for i, penalty in enumerate(penalties):
        block = view == i
        lam, vectors = np.linalg.eigh(gram[np.ix_(block, block)])
        whiten = vectors / np.sqrt(lam)
        size = int(block.sum())
        # The SVD of the factor resolves the penalty's null space to
        # eps * sqrt(|penalty|), where an eigendecomposition of the penalty
        # itself would give eps * |penalty|.
        _, singular, vt = np.linalg.svd(_penalty_factor(penalty, size) @ whiten)
        d = np.zeros(size)
        d[: singular.size] = singular**2
        rotation = vt.T
        keep = d < cutoff
        transform_i = np.zeros((view.size, int(keep.sum())))
        transform_i[block] = whiten @ rotation[:, keep]
        blocks.append((transform_i, d[keep]))
        kept_view.append(np.full(int(keep.sum()), i))
    transform = np.hstack([t for t, _ in blocks])
    d = np.concatenate([di for _, di in blocks])
    lhs = transform.T @ gram @ transform - np.diag(d) / 4
    return (lhs + lhs.T) / 2, transform, np.concatenate(kept_view)


def penalised_gram_ey_closed_form(
    gram: np.ndarray, view: np.ndarray, k: int, penalties: Sequence[float | np.ndarray]
) -> list[np.ndarray]:
    """Globally optimal penalised-EY coefficients from the stacked Gram matrix.

    Args:
        gram: Stacked Gram matrix, positive definite on each view's block.
        view: View index of each stacked column.
        k: Number of latent dimensions.
        penalties: One per view, as for :func:`penalised_gram_ey_gep`.

    Returns:
        Coefficients of shape (d_i, k) per view; components with no positive
        eigenvalue are zero.
    """
    lhs, transform, _ = _demmler_reinsch(gram, view, penalties)
    size = lhs.shape[0]
    mu, u = scipy.linalg.eigh(lhs, subset_by_index=(max(size - k, 0), size - 1))
    w = np.zeros((gram.shape[0], k))
    w[:, : len(mu)] = transform @ (u[:, ::-1] * np.sqrt(np.maximum(mu[::-1], 0.0)))
    return [w[view == i] for i in range(int(view.max()) + 1)]


def penalised_basis_ey_closed_form(
    bases: list[np.ndarray], k: int, penalties: Sequence[float | np.ndarray]
) -> list[np.ndarray]:
    """Globally optimal penalised-EY coefficients on fixed full-rank bases.

    Args:
        bases: Column-centred design matrix of each view.
        k: Number of latent dimensions.
        penalties: One per view, as for :func:`penalised_gram_ey_gep`.

    Returns:
        Coefficients of shape (n_basis_i, k) per view.
    """
    return penalised_gram_ey_closed_form(*_stacked_gram(bases), k, penalties)


def penalised_basis_ey_min_loss(
    bases: list[np.ndarray], k: int, penalties: Sequence[float | np.ndarray]
) -> float:
    """Minimum penalised-EY loss on fixed bases, ``-sum(mu**2)`` over the top ``k``."""
    lhs, _, _ = _demmler_reinsch(*_stacked_gram(bases), penalties)
    size = lhs.shape[0]
    mu = scipy.linalg.eigh(
        lhs, eigvals_only=True, subset_by_index=(max(size - k, 0), size - 1)
    )
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

    Split coefficients as $w = T a + N c$, with $T$ spanning the basis's row
    space and $N$ its null space. The embedding depends on $a$ alone, so the
    best $c$ minimises only the penalty $\|F(Ta + Nc)\|$, giving
    $w = L a$ with $L = T - N (F N)^{+} F T$: an equivalent full-rank
    problem with Gram $T^\top G T$ and penalty $(F L)^\top (F L)$.

    Numerically: the null basis from the Gram's eigendecomposition is
    refined once against the basis itself, since squaring the condition
    number contaminates it along weak row-space directions; the penalty
    least-squares problem is solved on $F$, not $N^\top R N$; and $F N$'s
    singular values count as zero below $\sqrt{\epsilon}\,\|F\|$.

    Args:
        basis: The basis, shape (n, d), supporting ``basis @ matrix``.
        gram: ``basis.T @ basis``, shape (d, d).
        penalty_factor: ``F`` with penalty ``F.T @ F``, shape (q, d).
        penalty_norm: Spectral norm of ``F``.

    Returns:
        ``(lift, row, reduced_factor)`` of shapes (d, r), (d, r) and (q, r):
        the reduced Gram is ``row.T @ gram @ row``, the reduced penalty
        ``reduced_factor.T @ reduced_factor``, and coefficients map back as
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
    return lift, row, factor_lift
