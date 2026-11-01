"""Column strengths of RankFreeJIVEEY as the penalty grows (one dataset per cell).

python diag_rankfree.py
"""

import time

import numpy as np
from common import N_SHARED, REGIMES
from rankfree import RankFreeJIVEEY
from simulate import simulate

for regime, rate in (("faint", 0.0), ("faint", 0.3)):
    d = REGIMES[regime]
    ds = simulate(0, n_shared=N_SHARED, missing_rate=rate, **d)
    for lam in (2.0, 5.0, 10.0, 20.0, 40.0):
        t = time.time()
        m = RankFreeJIVEEY(4, 5, 0.5, penalty=lam, max_iter=300).fit(ds.masked, seed=0, n_init=1)
        s = m.column_strengths()
        print(regime, rate, lam, "joint", np.round(s[:4], 2), "indiv0", np.round(s[4:9], 2), f"{time.time()-t:.1f}s", flush=True)
