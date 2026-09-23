"""Synthetic study: SSL gain versus redundancy (alpha) and copula fit."""

from __future__ import annotations

import pandas as pd
from joblib import Parallel, delayed

from experiment import evaluate
from synthetic import make_table

BUDGETS = [50, 100, 250, 1000]


def job(alpha: float, nonmonotone: bool, split: int) -> list[dict]:
    """Evaluate one synthetic table family at one split."""
    X, y = make_table(8000, alpha, nonmonotone=nonmonotone, seed=split)
    rows = evaluate(X, y, "reg", BUDGETS, reps=5, seed=split, dataset="synthetic")
    for r in rows:
        r.update(alpha=alpha, nonmonotone=nonmonotone)
    return rows


if __name__ == "__main__":
    jobs = [
        delayed(job)(a, nm, s)
        for a in [1.0, 0.75, 0.5, 0.25, 0.0]
        for nm in [False, True]
        for s in range(3)
    ]
    rows = sum(Parallel(n_jobs=4, verbose=5)(jobs), [])
    pd.DataFrame(rows).to_csv("results_synthetic.csv", index=False)
