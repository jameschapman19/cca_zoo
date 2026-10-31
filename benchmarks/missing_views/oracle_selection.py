"""Does MOFA+ find the shared factors that the selection rule discards?

The main benchmark keeps the two MOFA+ factors that explain the most variance in
every view. When the shared signal is a fraction of a percent of the variance
that rule can pick private factors. Here MOFA+ keeps all its factors and the true
shared factors are regressed on them, with R squared adjusted for the number of
factors, which is as good as a perfect choice of factors could do.

    python oracle_selection.py
"""

from __future__ import annotations

import numpy as np
import run
from methods import _mofa_once, _shared_factors, incomplete_gcca
from metrics import subspace_recovery
from simulate import simulate


def adjusted_r2(factors: np.ndarray, truth: np.ndarray) -> float:
    """Mean over true factors of the adjusted R squared of regressing on ``factors``."""
    n, k = factors.shape
    centred = factors - factors.mean(axis=0)
    scores = []
    for j in range(truth.shape[1]):
        y = truth[:, j] - truth[:, j].mean()
        residual = y - centred @ np.linalg.lstsq(centred, y, rcond=None)[0]
        r2 = 1 - np.sum(residual**2) / np.sum(y**2)
        scores.append(1 - (1 - r2) * (n - 1) / (n - k - 1))
    return float(np.mean(scores))


def main() -> None:
    """Print the comparison on complete data, four datasets per setting."""
    design = run.REGIMES["hard"]
    n_factors = run.N_SHARED + len(design["widths"]) * design["n_private"]
    for shared in design["shared"]:
        rows = []
        for seed in range(4):
            ds = simulate(
                seed,
                n_samples=design["n_samples"],
                widths=design["widths"],
                n_shared=run.N_SHARED,
                n_private=design["n_private"],
                shared=shared,
                private=design["private"],
            )
            ours, _ = incomplete_gcca(
                ds.masked, ds.observed, run.N_SHARED, seed, n_factors
            )
            default = _mofa_once(ds.masked, n_factors, seed, random_init=False)[0]
            selected = _shared_factors(default, ds.masked, run.N_SHARED)
            rows.append(
                (
                    adjusted_r2(ours, ds.shared_factors),
                    subspace_recovery(selected, ds.shared_factors),
                    adjusted_r2(default, ds.shared_factors),
                )
            )
        ours, selected, oracle = np.mean(rows, axis=0)
        print(
            f"shared {shared}: IncompleteGCCA {ours:.2f} | MOFA+ with the usual "
            f"selection {selected:.2f} | MOFA+ with all {n_factors} factors {oracle:.2f}"
        )


if __name__ == "__main__":
    main()
