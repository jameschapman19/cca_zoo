"""JIVEEY against the two-stage prototype, AJIVE and IncompleteGCCA on the simulator.

python compare_jiveey.py [--seeds 6]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from ey_shared_private import WhitenedSharedPrivateEY
from jiveey import JIVEEY

sys.path.insert(0, str(Path(__file__).parent.parent / "missing_views"))
from metrics import subspace_recovery  # noqa: E402
from simulate import simulate  # noqa: E402

from cca_zoo.linear import AJIVE, IncompleteGCCA  # noqa: E402

REGIMES = {
    "moderate": dict(
        n_samples=150, widths=(60, 80, 40), n_private=2, private=0.5, shared=0.1
    ),
    "faint": dict(
        n_samples=150, widths=(300, 400, 200), n_private=3, private=0.8, shared=0.02
    ),
    "fainter": dict(
        n_samples=150, widths=(300, 400, 200), n_private=3, private=0.8, shared=0.01
    ),
}
K = 2


def private_recovery(scores, truth, observed) -> float:
    """Mean over views of the individual-factor recovery on the samples that have the view."""
    return float(
        np.mean(
            [
                subspace_recovery(s[o], t[o])
                for s, t, o in zip(scores, truth, observed.T)
            ]
        )
    )


def run(regime: str, rate: float, seed: int) -> list[dict]:
    """Every method on one dataset."""
    design = REGIMES[regime]
    ds = simulate(seed, n_shared=K, missing_rate=rate, **design)
    masked, r, rows = ds.masked, design["n_private"], []

    def record(name, shared, private, start):
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
            )
        )

    for c in (0.2, 0.5, 0.8):
        t = time.perf_counter()
        model = JIVEEY(K, r, shrinkage=c).fit(masked, seed=seed)
        record(
            f"JIVEEY (c={c})",
            model.transform_joint(masked),
            model.transform_individual(masked),
            t,
        )
    t = time.perf_counter()
    two_stage = WhitenedSharedPrivateEY(K, r, shrinkage=0.5, weighting="equal").fit(
        masked, seed=seed
    )
    shared, private = two_stage.scores(masked)
    record("two-stage whitened EY (c=0.5)", shared, private, t)
    t = time.perf_counter()
    gcca = IncompleteGCCA(n_components=K, shrinkage=0.5).fit(masked)
    record("IncompleteGCCA", gcca.transform_shared(masked), None, t)
    t = time.perf_counter()
    filled = [
        np.where(o[:, None], v, np.nanmean(v, axis=0))
        for v, o in zip(masked, ds.observed.T)
    ]
    ajive = AJIVE(n_components=K, individual_ranks=r).fit(filled)
    record(
        "AJIVE (true ranks)",
        np.mean(ajive.transform(filled), axis=0),
        ajive.individual_scores_,
        t,
    )
    return rows


def main() -> None:
    """Print mean recovery by regime, missing rate and method."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=6)
    args = parser.parse_args()
    import pandas as pd

    rows = [
        row
        for regime in REGIMES
        for rate in (0.0, 0.3)
        for seed in range(args.seeds)
        for row in run(regime, rate, seed)
    ]
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(
        df.groupby(["regime", "rate", "method"])[["shared", "private", "seconds"]]
        .mean()
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
