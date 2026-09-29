"""The EY loss one coefficient at a time, shared by the coordinate-descent models.

Restricted to one coefficient, the EY loss is a quartic rather than the
quadratic of least squares; :func:`ey_quartic` gives its coefficients and
:func:`minimise_quartic` its exact penalised minimiser.
"""

from __future__ import annotations

import numpy as np

# Fractions of alpha at which the penalised models fit in turn, each stage
# warm-started from the last, as glmnet raises its penalty: all-zero weights
# are a local minimum once every view reaches them, so starting at the full
# penalty can collapse there.
PENALTY_PATH = (0.0, 0.1, 0.3, 1.0)


def ey_quartic(
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
    """Quartic coefficients ``(c4, c3, c2, c1)`` of the EY loss in one coefficient.

    The coefficient is component ``c`` of feature ``xj`` (squared norm ``a``)
    in a view with scores ``zi``, given every view's summed scores ``total``
    and the other views' score covariance ``v_other``; ``a0`` is
    ``1 / (m (n - 1))``.
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
    return p4, p3, p2 - 2 * a0 * a, p1 - 4 * a0 * x_s0c


def minimise_quartic(
    c4: float, c3: float, c2: float, c1: float, lasso: float, positive: bool = False
) -> float:
    """Global minimiser of ``c4 w^4 + c3 w^3 + c2 w^2 + c1 w + lasso |w|``.

    The best of the real stationary points on each side of zero and the kink
    at zero; ``c4 >= 0`` guarantees one exists. ``positive`` restricts the
    search to ``w >= 0``.
    """
    candidates = [0.0]
    branches = ((1.0, lasso),) if positive else ((1.0, lasso), (-1.0, -lasso))
    for sign, l1 in branches:
        roots = np.roots([4 * c4, 3 * c3, 2 * c2, c1 + l1])
        candidates.extend(
            float(r.real) for r in roots if abs(r.imag) < 1e-8 and sign * r.real > 0
        )

    def value(w: float) -> float:
        return c4 * w**4 + c3 * w**3 + c2 * w**2 + c1 * w + lasso * abs(w)

    return min(candidates, key=value)


def others_covariance(
    representations: list[np.ndarray], i: int, a0: float
) -> np.ndarray:
    """``a0`` times the summed score covariance of every view but view ``i``."""
    return np.asarray(
        sum(z.T @ z for j, z in enumerate(representations) if j != i) * a0
    )
