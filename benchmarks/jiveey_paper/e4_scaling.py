"""E4: fit time against the number of features and samples.

python e4_scaling.py
"""

from __future__ import annotations

import json
import time

import numpy as np
from common import HERE, JIVEEY, ajive_fit, mofa_joint_individual
from simulate import simulate

K, R = 2, 3


def timed(fn) -> float:
    """Wall-clock seconds of ``fn()``."""
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def main() -> None:
    """Time each method as features per view grow, then as samples grow."""
    rows = []
    for axis, grid in (("features", (100, 300, 1000, 3000, 10000)), ("samples", (100, 200, 400, 800))):
        for value in grid:
            width = value if axis == "features" else 1000
            n = 200 if axis == "features" else value
            ds = simulate(0, n_samples=n, widths=(width,) * 3, n_shared=K, n_private=R, shared=0.05, private=0.5, missing_rate=0.3)
            v = ds.masked
            rows.append(dict(axis=axis, value=value, method="JIVEEY", seconds=timed(lambda: JIVEEY(K, R, 0.5).fit(v, n_init=1))))
            rows.append(dict(axis=axis, value=value, method="AJIVE", seconds=timed(lambda: ajive_fit(v, ds.observed, K, R))))
            rows.append(dict(axis=axis, value=value, method="MOFA+", seconds=timed(lambda: mofa_joint_individual(v, ds.observed, K, R, 0))))
            print(rows[-3:], flush=True)
    (HERE / "output" / "e4.json").write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
