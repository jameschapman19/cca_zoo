"""Synthetic study: the cluster-assumption regime (label = latent segment)."""

from __future__ import annotations

import pandas as pd
from joblib import Parallel, delayed

from experiment import evaluate
from synthetic import make_clustered


def job(spread: float, split: int) -> list[dict]:
    """Evaluate one clustered table at one split."""
    X, y = make_clustered(8000, spread=spread, seed=split)
    rows = evaluate(X, y, "multi", [24, 48, 96, 192], reps=5, seed=split,
                    dataset="clustered")
    for r in rows:
        r.update(spread=spread)
    return rows


if __name__ == "__main__":
    jobs = [delayed(job)(s, sp) for s in [0.2, 0.35, 0.5] for sp in range(3)]
    rows = sum(Parallel(n_jobs=2)(jobs), [])
    pd.DataFrame(rows).to_csv("results_clustered.csv", index=False)
