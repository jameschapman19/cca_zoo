"""Run the missing-view benchmark and write one row per (setting, seed, method).

    python benchmarks/missing_views/run.py --out results.json [--seeds 10]

Needs the extras ``mofapy2`` for MOFA+.
"""

from __future__ import annotations

import argparse
import itertools
import json
from multiprocessing import Pool
from pathlib import Path

from methods import METHODS, timed
from metrics import imputation_r2, subspace_recovery
from simulate import simulate

SHARED = (0.05, 0.1)  # share of each feature's variance that is shared
MISSING_RATES = (0.0, 0.2, 0.4, 0.6)
MECHANISMS = ("mcar", "mnar")
DENSITIES = (1.0, 0.2)  # dense and sparse loadings
N_SHARED = 2
SIZE = {"n_samples": 150, "widths": (60, 80, 40), "private": 0.5}


def run_setting(setting: tuple) -> list[dict]:
    """Every method on one simulated dataset."""
    mechanism, shared, rate, density, seed = setting
    ds = simulate(
        seed,
        **SIZE,
        n_shared=N_SHARED,
        shared=shared,
        missing_rate=rate,
        mechanism=mechanism,
        density=density,
    )
    rows = []
    for name, method in METHODS.items():
        if name == "GCCA, complete cases" and ds.observed.all(axis=1).sum() < 30:
            continue  # too few complete samples to fit anything
        (latent, reconstruction), seconds = timed(
            method, ds.masked, ds.observed, N_SHARED, seed
        )
        rows.append(
            {
                "mechanism": mechanism,
                "shared": shared,
                "missing_rate": rate,
                "density": density,
                "seed": seed,
                "method": name,
                "recovery": subspace_recovery(latent, ds.shared_factors),
                "imputation_r2": (
                    imputation_r2(reconstruction, ds.shared_signal, ds.observed)
                    if rate > 0
                    else None
                ),
                "seconds": seconds,
            }
        )
    return rows


def main() -> None:
    """Parse the arguments and run every setting."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    settings = [
        (mechanism, shared, rate, density, seed)
        for mechanism, shared, rate, density, seed in itertools.product(
            MECHANISMS, SHARED, MISSING_RATES, DENSITIES, range(args.seeds)
        )
        if not (mechanism == "mnar" and rate == 0.0)  # the same data as MCAR
    ]
    with Pool(args.workers) as pool:
        rows = []
        for done, result in enumerate(pool.imap_unordered(run_setting, settings), 1):
            rows.extend(result)
            print(f"{done}/{len(settings)}", flush=True)
    args.out.write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
