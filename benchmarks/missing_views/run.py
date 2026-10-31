"""Run the missing-view benchmark and write one row per (setting, seed, method).

    python benchmarks/missing_views/run.py --out results.json [--seeds 10]

Needs the extras ``mofapy2`` for MOFA+.
"""

from __future__ import annotations

import argparse
import itertools
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

from methods import METHODS, timed
from metrics import imputation_r2, subspace_recovery
from simulate import simulate

N_SHARED = 2
MECHANISMS = ("mcar", "mnar")
DENSITIES = (1.0, 0.2)  # dense and sparse loadings
# Share of each feature's variance that is shared and private, and the problem
# size. "hard" has wide views, many private factors and a faint shared signal.
REGIMES = {
    "base": {
        "shared": (0.05, 0.1),
        "rates": (0.0, 0.2, 0.4, 0.6),
        "private": 0.5,
        "n_private": 2,
        "n_samples": 150,
        "widths": (60, 80, 40),
    },
    "hard": {
        "shared": (0.005, 0.01, 0.02),
        "rates": (0.0, 0.3, 0.5),
        "private": 0.8,
        "n_private": 3,
        "n_samples": 150,
        "widths": (300, 400, 200),
    },
}


def run_setting(setting: tuple, methods: tuple[str, ...]) -> list[dict]:
    """Every method on one simulated dataset."""
    regime, mechanism, shared, rate, density, seed = setting
    design = REGIMES[regime]
    ds = simulate(
        seed,
        n_samples=design["n_samples"],
        widths=design["widths"],
        n_shared=N_SHARED,
        n_private=design["n_private"],
        shared=shared,
        private=design["private"],
        missing_rate=rate,
        mechanism=mechanism,
        density=density,
    )
    n_factors = N_SHARED + len(design["widths"]) * design["n_private"]
    rows = []
    for name in methods:
        method = METHODS[name]
        if name == "GCCA, complete cases" and ds.observed.all(axis=1).sum() < 30:
            continue  # too few complete samples to fit anything
        (latent, reconstruction), seconds = timed(
            method, ds.masked, ds.observed, N_SHARED, seed, n_factors
        )
        rows.append(
            {
                "regime": regime,
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
    parser.add_argument("--regime", choices=list(REGIMES), default="base")
    parser.add_argument(
        "--methods", nargs="+", choices=list(METHODS), default=list(METHODS)
    )
    args = parser.parse_args()

    design = REGIMES[args.regime]
    settings = [
        (args.regime, mechanism, shared, rate, density, seed)
        for mechanism, shared, rate, density, seed in itertools.product(
            MECHANISMS, design["shared"], design["rates"], DENSITIES, range(args.seeds)
        )
        if not (mechanism == "mnar" and rate == 0.0)  # the same data as MCAR
    ]
    with Pool(args.workers) as pool:
        rows = []
        for done, result in enumerate(
            pool.imap_unordered(
                partial(run_setting, methods=tuple(args.methods)), settings
            ),
            1,
        ):
            rows.extend(result)
            print(f"{done}/{len(settings)}", flush=True)
    args.out.write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
