"""Real-table study: label-scarce SSL gains on the reachable datasets."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parents[2] / "benchmarks" / "canonical_boosting"))
from run import DATASETS, load  # noqa: E402

from experiment import evaluate  # noqa: E402

BUDGETS = [32, 64, 128, 256, 512]


OUT = Path(__file__).parent / "results_real"


def job(name: str, split: int) -> None:
    """Evaluate one dataset at one pool/test split; write its rows to disk."""
    path = OUT / f"{name}_{split}.csv"
    if path.exists():
        return
    X, y, task = load(name)
    rows = evaluate(X, y, task, BUDGETS, reps=5, seed=split, dataset=name)
    for r in rows:
        r.update(task=task, n=len(X), d=X.shape[1])
    pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    names = sys.argv[1:] or DATASETS
    for name in names:
        load(name)
    OUT.mkdir(exist_ok=True)
    Parallel(n_jobs=4, verbose=5)(delayed(job)(n, s) for n in names for s in range(3))
    pd.concat(map(pd.read_csv, sorted(OUT.glob("*.csv")))).to_csv(
        "results_real.csv", index=False
    )
