"""Does the divisor of the pairwise covariances explain the gap to mean-filled AJIVE?

Variants, all with the true ranks: the pairwise divisors n_ij - 1 (current); n - 1 for every
block, which is what mean-filling does to the cross-covariances; n - 1 off the diagonal only.

python diag_weighting.py
"""

from __future__ import annotations

import numpy as np
from common import JIVEEY, N_SHARED, REGIMES, subspace_recovery
from simulate import simulate


class Variant(JIVEEY):
    """JIVEEY with a different divisor for the covariance blocks."""

    def __init__(self, *args, mode: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode

    def _prepare(self, views):
        super()._prepare(views)
        n = len(views[0]) - 1.0
        if self.mode == "total":
            self.divisor_ = np.full_like(self.divisor_, n)
        elif self.mode == "offdiag":
            d = np.full_like(self.divisor_, n)
            np.fill_diagonal(d, np.diag(self.divisor_))
            self.divisor_ = d


CELLS = (("moderate", 0.5), ("faint", 0.3), ("faint", 0.5), ("fainter", 0.3))
for regime, rate in CELLS:
    design = REGIMES[regime]
    res: dict[str, list[float]] = {}
    for seed in range(10):
        ds = simulate(seed, n_shared=N_SHARED, missing_rate=rate, **design)
        for mode in ("pair", "total", "offdiag"):
            for c in (0.5, 0.8):
                m = Variant(N_SHARED, design["n_private"], c, mode=mode).fit(ds.masked, seed=seed)
                res.setdefault(f"{mode} c={c}", []).append(
                    subspace_recovery(m.transform_joint(ds.masked), ds.shared_factors)
                )
    print(regime, f"{rate:.0%}", {k: round(float(np.mean(v)), 2) for k, v in res.items()}, flush=True)
