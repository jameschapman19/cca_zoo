"""TCCA: PARAFAC of the whitened cross-moment tensor."""

from __future__ import annotations

import numpy as np

from cca_zoo.linear import TCCA
from tests._helpers import ordered_views


def _third_moment(scores: list[np.ndarray]) -> float:
    """TCCA's objective for one component: the scores' standardised co-moment."""
    standardised = [(z - z.mean()) / z.std() for z in (s[:, 0] for s in scores)]
    return float(abs(np.mean(np.prod(standardised, axis=0))))


def test_no_random_start_beats_the_svd_start() -> None:
    """PARAFAC has local optima; random starts find none better than the SVD start."""
    views = ordered_views(0, 500, (6, 5, 4))
    svd = _third_moment(TCCA(1).fit(views).transform(views))
    starts = [
        _third_moment(
            TCCA(1, init="random", random_state=seed).fit(views).transform(views)
        )
        for seed in range(10)
    ]
    assert max(starts) <= svd + 1e-9
    assert np.isclose(max(starts), svd)
