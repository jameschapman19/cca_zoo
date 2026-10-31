"""Is LeJEPA's digits result a lambda artefact? LeJEPA (V-stat SIGReg) at several lambdas, batch 256.

Run:  python scripts/digits_lambda.py   Writes results/digits_lambda.npz.
"""

from __future__ import annotations

import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ieye import ssl

import digits_ssl

ROOT = Path(__file__).resolve().parents[1]
LAMBDAS = [0.01, 0.2, 0.5, 0.8]
SEEDS = [500, 501, 502]


def run(job: tuple[float, int]) -> tuple[float, list[float]]:
    lam, seed = job
    ssl.LAMBDA = lam
    return lam, digits_ssl.run(("LeJEPA (SIGReg)", 256, seed))[1]


if __name__ == "__main__":
    jobs = [(lam, s) for lam in LAMBDAS for s in SEEDS]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with Pool(4) as pool:
            out = pool.map(run, jobs, chunksize=1)
    acc = np.array([[o[1] for o in out if o[0] == lam] for lam in LAMBDAS])  # (lam, seed, probe)
    np.savez(ROOT / "results/digits_lambda.npz", accuracy=acc, lambdas=LAMBDAS)
    for lam, a in zip(LAMBDAS, acc):
        print(f"lambda={lam}: linear {a[:, 0].mean():.3f}  knn {a[:, 1].mean():.3f}")
