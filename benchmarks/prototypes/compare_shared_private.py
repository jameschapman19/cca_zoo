"""Does the Eckart-Young shared/private fit recover shared and private factors?

Compared with AJIVE (given the true ranks) and IncompleteGCCA (shared only) on the
simulator, complete and with views missing. AJIVE needs complete data, so it gets
the views with missing samples set to their mean.

    python compare_shared_private.py [--seeds 4]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from ey_shared_private import SharedPrivateEY, WhitenedSharedPrivateEY

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


def private_recovery(
    scores: list[np.ndarray], truth: list[np.ndarray], observed: np.ndarray
) -> float:
    """Mean over views of the subspace recovery of the private factors."""
    values = []
    for s, t, o in zip(scores, truth, observed.T):
        values.append(subspace_recovery(s[o], t[o]))
    return float(np.mean(values))


def run(regime: str, rate: float, seed: int) -> list[dict]:
    """Every method on one dataset."""
    design = REGIMES[regime]
    ds = simulate(seed, n_shared=K, missing_rate=rate, **design)
    masked, r = ds.masked, design["n_private"]
    rows = []

    def record(
        name: str, shared: np.ndarray, private: list[np.ndarray] | None, seconds: float
    ) -> None:
        rows.append(
            {
                "regime": regime,
                "rate": rate,
                "seed": seed,
                "method": name,
                "shared": subspace_recovery(shared, ds.shared_factors),
                "private": None
                if private is None
                else private_recovery(private, ds.private_factors, ds.observed),
                "seconds": seconds,
            }
        )

    t = time.perf_counter()
    gcca = IncompleteGCCA(n_components=K, shrinkage=0.5).fit(masked)
    latent = gcca.transform_shared(masked)
    record("IncompleteGCCA", latent, None, time.perf_counter() - t)

    t = time.perf_counter()
    random_start = SharedPrivateEY(K, r).fit(masked, seed=seed)
    shared, private = random_start.scores(masked)
    record("EY shared/private (random start)", shared, private, time.perf_counter() - t)

    t = time.perf_counter()
    seeded = SharedPrivateEY(K, r).fit(masked, latent=latent / latent.std(axis=0))
    shared, private = seeded.scores(masked)
    record(
        "EY shared/private (from IncompleteGCCA)",
        shared,
        private,
        time.perf_counter() - t,
    )

    for c in (0.2, 0.5, 0.8):
        t = time.perf_counter()
        whitened = WhitenedSharedPrivateEY(K, r, shrinkage=c).fit(masked, seed=seed)
        shared, private = whitened.scores(masked)
        record(
            f"EY shared/private, whitened (c={c})",
            shared,
            private,
            time.perf_counter() - t,
        )

    t = time.perf_counter()
    filled = [
        np.where(o[:, None], v, np.nanmean(v, axis=0))
        for v, o in zip(masked, ds.observed.T)
    ]
    ajive = AJIVE(n_components=K, individual_ranks=r).fit(filled)
    shared = np.mean(ajive.transform(filled), axis=0)
    record(
        "AJIVE (true ranks)", shared, ajive.individual_scores_, time.perf_counter() - t
    )
    return rows


def main() -> None:
    """Print mean recovery by regime, missing rate and method."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=4)
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
    table = (
        df.groupby(["regime", "rate", "method"])[["shared", "private", "seconds"]]
        .mean()
        .round(2)
    )
    print(table.to_string())


if __name__ == "__main__":
    main()
