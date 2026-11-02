"""E7: automatic AJIVE-EM against what practitioners use, on seeds never used before (200+).

Every method but MOFA+ chooses its own ranks; MOFA+ is given the true factor counts.

python e7_practical.py [--seeds 20] [--first 200] [--workers 4]
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool

import numpy as np
from ajive_em import ajive_em_observed_ranks
from common import (
    HERE,
    N_SHARED,
    REGIMES,
    ajive_fit,
    mean_filled,
    mofa_joint_individual,
    private_recovery,
    subspace_recovery,
)
from e5_fresh import VARIANTS
from simulate import simulate


def run(job: tuple[str, str, float, int]) -> list[dict]:
    """The methods on one dataset."""
    variant, regime, rate, seed = job
    design = REGIMES[regime]
    ds = simulate(seed, n_shared=N_SHARED, missing_rate=rate, **design, **VARIANTS[variant])
    views, r, rows = ds.masked, design["n_private"], []

    def record(name, shared, private, start, **extra):
        rows.append(
            dict(
                variant=variant, regime=regime, rate=rate, seed=seed, method=name,
                shared=subspace_recovery(shared, ds.shared_factors),
                private=None if private is None else private_recovery(private, ds.private_factors, ds.observed),
                seconds=time.perf_counter() - start, **extra,
            )
        )

    def miss(name, start):
        rows.append(dict(variant=variant, regime=regime, rate=rate, seed=seed, method=name, shared=0.0, private=0.0, seconds=time.perf_counter() - start, k=0))

    t = time.perf_counter()
    a, done, _, _ = ajive_em_observed_ranks(views, ds.observed, seed)
    if a is None:
        miss("AJIVE-EM (automatic)", t)
    else:
        record("AJIVE-EM (automatic)", np.mean(a.transform(done), axis=0), a.individual_scores_, t, k=int(a.joint_scores_.shape[1]))

    t = time.perf_counter()
    try:
        a = ajive_fit(views, ds.observed)
    except ValueError:
        miss("AJIVE, mean-filled (automatic)", t)
    else:
        record("AJIVE, mean-filled (automatic)", np.mean(a.transform(mean_filled(views, ds.observed)), axis=0), a.individual_scores_, t, k=int(a.joint_scores_.shape[1]))

    t = time.perf_counter()
    s, p = mofa_joint_individual(views, ds.observed, N_SHARED, r, seed)
    record("MOFA+ (true ranks)", s, p, t)
    return rows


def main() -> None:
    """Run the grid and save it as it goes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--first", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    seeds = range(args.first, args.first + args.seeds)
    jobs = [("base", g, rate, s) for g in REGIMES for rate in (0.0, 0.3, 0.5) for s in seeds]
    jobs += [("sparse_mnar", g, rate, s) for g in REGIMES for rate in (0.3, 0.5) for s in seeds]
    path = HERE / "output" / "e7.json"
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
