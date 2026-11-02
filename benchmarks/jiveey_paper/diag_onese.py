"""Does the one-standard-error rule fix the rank selection with views missing?

python diag_onese.py
"""

from multiprocessing import Pool

import numpy as np
from common import JIVEEY, N_SHARED, REGIMES, select_ranks, subspace_recovery
from simulate import simulate

CELLS = (("faint", 0.3), ("faint", 0.5), ("fainter", 0.3), ("moderate", 0.3), ("faint", 0.0))


def run(job):
    regime, rate, seed, one_se = job
    d = REGIMES[regime]
    ds = simulate(seed, n_shared=N_SHARED, missing_rate=rate, **d)
    k, r = select_ranks(ds.masked, 0.5, seed=seed, one_se=one_se)
    m = JIVEEY(k, r, 0.5).fit(ds.masked, seed=seed)
    return regime, rate, one_se, k, r, subspace_recovery(m.transform_joint(ds.masked), ds.shared_factors)


if __name__ == "__main__":
    jobs = [(g, r, s, o) for g, r in CELLS for s in range(8) for o in (False, True)]
    with Pool(4) as p:
        out = p.map(run, jobs)
    for g, r in CELLS:
        for o in (False, True):
            sel = [x for x in out if x[0] == g and x[1] == r and x[2] == o]
            print(g, r, "1-SE" if o else "min", "K", [x[3] for x in sel], "r", [x[4] for x in sel], "recovery", round(float(np.mean([x[5] for x in sel])), 2))
