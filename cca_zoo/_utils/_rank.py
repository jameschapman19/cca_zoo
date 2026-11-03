"""Rank estimates for noisy low-rank matrices."""

from __future__ import annotations

import numpy as np


def optimal_hard_threshold_rank(singular_values: np.ndarray, n: int, p: int) -> int:
    """Singular values above Gavish and Donoho's hard threshold, noise unknown.

    The threshold is ``omega(beta)`` times the median singular value, with
    ``beta`` the aspect ratio of the ``n`` by ``p`` matrix and ``omega`` their
    polynomial approximation to the optimal coefficient.

    Args:
        singular_values: Singular values of the centred ``n`` by ``p`` matrix.
        n: Number of rows.
        p: Number of columns.

    Returns:
        The number of singular values above the threshold.

    References:
        Gavish, M. and Donoho, D. L. (2014). The optimal hard threshold for
        singular values is 4/sqrt(3). IEEE Transactions on Information Theory,
        60(8), 5040-5053.
    """
    beta = min(n, p) / max(n, p)
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    return int(np.sum(singular_values > omega * np.median(singular_values)))
