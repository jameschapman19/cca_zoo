"""The held-out cross-view correlation rank rule on fresh seeds (100+), against the one-SE grid rule.

python diag_ranks2.py
"""

from multiprocessing import Pool

import numpy as np
from common import JIVEEY, N_SHARED, REGIMES, select_ranks, subspace_recovery


def symmetric(a, b):
    qa = np.linalg.qr(a - a.mean(0))[0]
    qb = np.linalg.qr(b - b.mean(0))[0]
    return float(2 * np.sum(np.linalg.svd(qa.T @ qb, compute_uv=False) ** 2) / (qa.shape[1] + qb.shape[1]))

from ranks2 import select_ranks2
from simulate import simulate

CELLS = (("faint", 0.3), ("faint", 0.5), ("fainter", 0.3), ("moderate", 0.3), ("moderate", 0.5), ("faint", 0.0))


def run(job):
    regime, rate, seed, rule = job
    ds = simulate(seed, n_shared=N_SHARED, missing_rate=rate, **REGIMES[regime])
    k, r = select_ranks2(ds.masked, seed=seed) if rule == "xview" else select_ranks(ds.masked, 0.5, seed=seed)
    m = JIVEEY(k, r, 0.5).fit(ds.masked, seed=seed)
    z = m.transform_joint(ds.masked)
    return regime, rate, rule, k, r, subspace_recovery(z, ds.shared_factors), symmetric(z, ds.shared_factors)


if __name__ == "__main__":
    jobs = [(g, r, s, rule) for g, r in CELLS for s in range(100, 108) for rule in ("xview", "grid")]
    with Pool(4) as p:
        out = p.map(run, jobs)
    for g, r in CELLS:
        for rule in ("grid", "xview"):
            sel = [x for x in out if x[0] == g and x[1] == r and x[2] == rule]
            print(g, r, rule, "K", [x[3] for x in sel], "r", [x[4] for x in sel], "contain", round(float(np.mean([x[5] for x in sel])), 2), "symmetric", round(float(np.mean([x[6] for x in sel])), 2))
