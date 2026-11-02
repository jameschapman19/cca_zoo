"""E5: the recovery grid on fresh seeds, with the symmetric overlap metric and an AJIVE that completes
missing views by low-rank EM, plus a second design (sparse loadings, missingness driven by the first
shared factor) with no cross-validation.

python e5_fresh.py [--seeds 20] [--first 100] [--workers 4]
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool

import numpy as np
from common import (
    HERE,
    JIVEEY,
    N_SHARED,
    REGIMES,
    IncompleteGCCA,
    ajive_fit,
    em_complete,
    mean_filled,
    mofa_joint_individual,
    private_recovery,
    select_ranks,
    select_shrinkage,
    subspace_recovery,
)
from simulate import simulate

VARIANTS = {
    "base": dict(),
    "sparse_mnar": dict(density=0.3, mechanism="mnar"),
}


def run(job: tuple[str, str, float, int]) -> list[dict]:
    """Every method on one dataset."""
    variant, regime, rate, seed = job
    design = REGIMES[regime]
    ds = simulate(seed, n_shared=N_SHARED, missing_rate=rate, **design, **VARIANTS[variant])
    views, r, rows = ds.masked, design["n_private"], []
    total = N_SHARED + len(views) * r

    def record(name, shared, private, start, **extra):
        rows.append(
            dict(
                variant=variant, regime=regime, rate=rate, seed=seed, method=name,
                shared=subspace_recovery(shared, ds.shared_factors),
                private=None if private is None else private_recovery(private, ds.private_factors, ds.observed),
                seconds=time.perf_counter() - start, **extra,
            )
        )

    t = time.perf_counter()
    m = JIVEEY(N_SHARED, r, 0.5).fit(views, seed=seed)
    record("JIVEEY (true ranks)", m.transform_joint(views), m.transform_individual(views), t)

    if variant == "base":
        t = time.perf_counter()
        k, rr = select_ranks(views, 0.5, seed=seed)
        c = select_shrinkage(views, k, rr, seed=seed)
        m = JIVEEY(k, rr, c).fit(views, seed=seed)
        record("JIVEEY (selected)", m.transform_joint(views), m.transform_individual(views), t, k=k, r=rr, c=c)

    t = time.perf_counter()
    m = JIVEEY(N_SHARED, 0, 0.5).fit(views, seed=seed)
    record("CCA-EY (no individual)", m.transform_joint(views), None, t)

    t = time.perf_counter()
    g = IncompleteGCCA(n_components=N_SHARED, shrinkage=0.5).fit(views)
    record("IncompleteGCCA", g.transform_shared(views), None, t)

    t = time.perf_counter()
    a = ajive_fit(views, ds.observed, N_SHARED, r)
    record("AJIVE, mean-filled (true ranks)", np.mean(a.transform(mean_filled(views, ds.observed)), axis=0), a.individual_scores_, t)

    t = time.perf_counter()
    done = em_complete(views, ds.observed, total)
    a = ajive_fit(done, np.ones_like(ds.observed), N_SHARED, r)
    record("AJIVE, EM-completed (true ranks)", np.mean(a.transform(done), axis=0), a.individual_scores_, t)

    t = time.perf_counter()
    s, p = mofa_joint_individual(views, ds.observed, N_SHARED, r, seed)
    record("MOFA+ (true ranks)", s, p, t)
    return rows


def main() -> None:
    """Run the grid and save it as it goes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--first", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    seeds = range(args.first, args.first + args.seeds)
    jobs = [("base", g, rate, s) for g in REGIMES for rate in (0.0, 0.3, 0.5) for s in seeds]
    jobs += [("sparse_mnar", g, rate, s) for g in REGIMES for rate in (0.3, 0.5) for s in seeds]
    path = HERE / "output" / "e5.json"
    rows: list[dict] = json.loads(path.read_text()) if path.exists() else []
    done = {(r["variant"], r["regime"], r["rate"], r["seed"]) for r in rows}
    jobs = [j for j in jobs if j not in done]
    with Pool(args.workers) as pool:
        for n, out in enumerate(pool.imap_unordered(run, jobs), 1):
            rows.extend(out)
            path.write_text(json.dumps(rows))
            print(f"{n}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
