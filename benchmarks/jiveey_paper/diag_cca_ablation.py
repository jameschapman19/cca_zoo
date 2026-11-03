"""Which ingredient of AJIVE-EM matters for a CCA-like method: rank truncation, EM completion, or both?

All variants are IncompleteGCCA with the true joint rank 2, on seeds 300-309.

python diag_cca_ablation.py
"""

from multiprocessing import Pool

import numpy as np
from ajive_em import observed_signal_ranks
from common import N_SHARED, REGIMES, IncompleteGCCA, em_complete, subspace_recovery
from simulate import simulate

CELLS = (("faint", 0.3), ("faint", 0.5), ("fainter", 0.3), ("moderate", 0.3), ("moderate", 0.5))


def truncate(views, observed, ranks):
    """Project each view onto the top PCA directions of its observed rows; NaN rows stay NaN."""
    out = []
    for v, o, s in zip(views, observed.T, ranks):
        x = v[o] - v[o].mean(0)
        vt = np.linalg.svd(x, full_matrices=False)[2][:s]
        z = np.full((len(v), s), np.nan)
        z[o] = x @ vt.T
        out.append(z)
    return out


def run(job):
    regime, rate, seed = job
    ds = simulate(seed, n_shared=N_SHARED, missing_rate=rate, **REGIMES[regime])
    v, o = ds.masked, ds.observed
    sig = observed_signal_ranks(v, o)
    res = {}
    res["IncompleteGCCA, shrinkage 0.5"] = IncompleteGCCA(n_components=2, shrinkage=0.5).fit(v).transform_shared(v)
    res["truncated only"] = IncompleteGCCA(n_components=2, shrinkage=0.0).fit(truncate(v, o, sig)).transform_shared(truncate(v, o, sig))
    done = em_complete(v, o, sum(sig))
    res["EM only (shrinkage 0.5)"] = IncompleteGCCA(n_components=2, shrinkage=0.5).fit(done).transform_shared(done)
    dt = truncate(done, np.ones_like(o), sig)
    res["EM + truncated"] = IncompleteGCCA(n_components=2, shrinkage=0.0).fit(dt).transform_shared(dt)
    return regime, rate, {k: subspace_recovery(z, ds.shared_factors) for k, z in res.items()}


if __name__ == "__main__":
    jobs = [(g, r, s) for g, r in CELLS for s in range(300, 310)]
    with Pool(4) as p:
        out = p.map(run, jobs)
    for g, r in CELLS:
        sel = [x[2] for x in out if x[0] == g and x[1] == r]
        print(g, r, {k: round(float(np.mean([s[k] for s in sel])), 2) for k in sel[0]})
