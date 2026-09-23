"""Print summary tables from the result CSVs."""

from __future__ import annotations

import sys

import pandas as pd

pd.set_option("display.width", 200)


def synthetic(path: str = "results_synthetic.csv") -> None:
    """Mean test R^2 minus raw-feature R^2, same probe, per alpha and budget."""
    d = pd.read_csv(path)
    m = d.groupby(["nonmonotone", "alpha", "budget", "probe", "features"]).score.mean()
    base = m.xs("raw", level="features")
    gain = (m - base.reindex(m.droplevel("features").index).values).unstack("features")
    for nm in [False, True]:
        print(f"\n=== nonmonotone={nm}: raw R^2 ===")
        print(base.xs(nm).unstack("probe").round(3))
        print(f"--- gain over raw (same probe) ---")
        print(gain.xs(nm).drop(columns="raw").round(3))


def real(path: str = "results_real.csv") -> None:
    """Mean score minus raw-feature score, same probe, per dataset and budget."""
    d = pd.read_csv(path)
    m = d.groupby(["dataset", "budget", "probe", "features"]).score.mean()
    base = m.xs("raw", level="features")
    gain = (m - base.reindex(m.droplevel("features").index).values).unstack("features")
    print(base.unstack("probe").round(3))
    print(gain.drop(columns="raw").round(3))
    print("\n=== mean gain over datasets, by budget ===")
    print(gain.drop(columns="raw").groupby(["budget", "probe"]).mean().round(4))


if __name__ == "__main__":
    {"synthetic": synthetic, "real": real}[sys.argv[1]]()
