"""E1: recovery of the joint and individual factors, 20 datasets per cell.

python e1_recovery.py [--seeds 20] [--workers 4]
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool

import numpy as np
from common import (
    HERE,
    N_SHARED,
    REGIMES,
    IncompleteGCCA,
    JIVEEY,
    ajive_fit,
    mean_filled,
    mofa_joint_individual,
    private_recovery,
    select_ranks,
    select_shrinkage,
    subspace_recovery,
)
from simulate import simulate


def run(job: tuple[str, float, int]) -> list[dict]:
    """Every method on one dataset."""
    regime, rate, seed = job
    design = REGIMES[regime]
    ds = simulate(seed, n_shared=N_SHARED, missing_rate=rate, **design)
    views, r, rows = ds.masked, design["n_private"], []

    def record(name, shared, private, start, **extra):
        rows.append(
            dict(
                regime=regime,
                rate=rate,
                seed=seed,
                method=name,
                shared=subspace_recovery(shared, ds.shared_factors),
                private=None
                if private is None
                else private_recovery(private, ds.private_factors, ds.observed),
                seconds=time.perf_counter() - start,
                **extra,
            )
        )

    t = time.perf_counter()
    m = JIVEEY(N_SHARED, r, 0.5).fit(views, seed=seed)
    record("JIVEEY (true ranks)", m.transform_joint(views), m.transform_individual(views), t)

    t = time.perf_counter()
    k, rr = select_ranks(views, 0.5, seed=seed)
    c = select_shrinkage(views, k, rr, seed=seed)
    m = JIVEEY(k, rr, c).fit(views, seed=seed)
    record(
        "JIVEEY (selected)",
        m.transform_joint(views),
        m.transform_individual(views),
        t,
        k=k,
        r=rr,
        c=c,
    )

    t = time.perf_counter()
    m = JIVEEY(N_SHARED, 0, 0.5).fit(views, seed=seed)
    record("CCA-EY (no individual)", m.transform_joint(views), None, t)

    t = time.perf_counter()
    g = IncompleteGCCA(n_components=N_SHARED, shrinkage=0.5).fit(views)
    record("IncompleteGCCA", g.transform_shared(views), None, t)

    t = time.perf_counter()
    a = ajive_fit(views, ds.observed, N_SHARED, r)
    record(
        "AJIVE (true ranks)",
        np.mean(a.transform(mean_filled(views, ds.observed)), axis=0),
        a.individual_scores_,
        t,
    )
    t = time.perf_counter()
    try:
        a = ajive_fit(views, ds.observed)
    except ValueError:  # AJIVE found no joint structure: report that as a total miss
        rows.append(
            dict(regime=regime, rate=rate, seed=seed, method="AJIVE (auto ranks)",
                 shared=0.0, private=0.0, seconds=time.perf_counter() - t, k=0)
        )
    else:
        record(
            "AJIVE (auto ranks)",
            np.mean(a.transform(mean_filled(views, ds.observed)), axis=0),
            a.individual_scores_,
            t,
            k=int(a.joint_scores_.shape[1]),
        )

    t = time.perf_counter()
    s, p = mofa_joint_individual(views, ds.observed, N_SHARED, r, seed)
    record("MOFA+", s, p, t)
    return rows


def main() -> None:
    """Run the grid and save it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    jobs = [
        (g, rate, s)
        for g in REGIMES
        for rate in (0.0, 0.3, 0.5)
        for s in range(args.seeds)
    ]
    path = HERE / "output" / "e1.json"
    rows: list[dict] = json.loads(path.read_text()) if path.exists() else []
    done = {(r["regime"], r["rate"], r["seed"]) for r in rows}
    jobs = [j for j in jobs if j not in done]
    with Pool(args.workers) as pool:
        for done, out in enumerate(pool.imap_unordered(run, jobs), 1):
            rows.extend(out)
            (HERE / "output" / "e1.json").write_text(json.dumps(rows))
            print(f"{done}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
