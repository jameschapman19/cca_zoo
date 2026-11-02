"""Does 5-fold rank selection fix the joint rank at 50% missing?

python diag_folds.py
"""

from multiprocessing import Pool

import numpy as np
from common import JIVEEY, N_SHARED, REGIMES, select_ranks, subspace_recovery
from simulate import simulate


def run(job):
    regime, rate, seed, folds = job
    d = REGIMES[regime]
    ds = simulate(seed, n_shared=N_SHARED, missing_rate=rate, **d)
    k, r = select_ranks(ds.masked, 0.5, seed=seed, n_splits=folds)
    m = JIVEEY(k, r, 0.5).fit(ds.masked, seed=seed)
    return regime, rate, folds, k, subspace_recovery(m.transform_joint(ds.masked), ds.shared_factors)


if __name__ == "__main__":
    jobs = [(g, r, s, f) for g, r in (("faint", 0.5), ("fainter", 0.3)) for s in range(8) for f in (3, 5)]
    with Pool(3) as p:
        out = p.map(run, jobs)
    for g, r in (("faint", 0.5), ("fainter", 0.3)):
        for f in (3, 5):
            sel = [o for o in out if o[0] == g and o[1] == r and o[2] == f]
            print(g, r, f"{f}-fold", "ks", [o[3] for o in sel], "recovery", round(float(np.mean([o[4] for o in sel])), 2))
