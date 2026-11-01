"""Does JIVEEY recover how much of each view's variance is joint, individual and noise?

The simulator fixes the truth: each feature's variance is ``shared`` joint, ``private`` individual
and the rest noise. Compared with AJIVE's own variance accounting.

    python jiveey_variance.py [--seeds 6]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from jiveey import JIVEEY

sys.path.insert(0, str(Path(__file__).parent.parent / "missing_views"))
from simulate import simulate  # noqa: E402

from cca_zoo.linear import AJIVE  # noqa: E402

DESIGNS = {
    "moderate": dict(
        n_samples=150, widths=(60, 80, 40), n_private=2, private=0.5, shared=0.1
    ),
    "faint": dict(
        n_samples=150, widths=(300, 400, 200), n_private=3, private=0.8, shared=0.02
    ),
}


def main() -> None:
    """Print the estimated joint, individual and noise shares against the true ones."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=6)
    args = parser.parse_args()
    print(
        f"{'regime':<9}{'views missing':>14}{'method':>9}{'joint':>8}{'individual':>12}{'noise':>8}"
    )
    for name, design in DESIGNS.items():
        truth = (
            design["shared"],
            design["private"],
            1 - design["shared"] - design["private"],
        )
        print(
            f"{name:<9}{'':>14}{'truth':>9}{truth[0]:>8.2f}{truth[1]:>12.2f}{truth[2]:>8.2f}"
        )
        for rate in (0.0, 0.3):
            ours, theirs = [], []
            for seed in range(args.seeds):
                ds = simulate(seed, n_shared=2, missing_rate=rate, **design)
                model = JIVEEY(2, design["n_private"], shrinkage=0.5).fit(
                    ds.masked, seed=seed
                )
                v = model.variance_explained(ds.masked)
                ours.append(
                    [
                        v["joint_total"].mean(),
                        v["individual_total"].mean(),
                        v["noise"].mean(),
                    ]
                )
                if rate == 0.0:
                    a = AJIVE(n_components=2, individual_ranks=design["n_private"]).fit(
                        ds.views
                    )
                    theirs.append(
                        [
                            a.joint_variance_explained_.sum(1).mean(),
                            np.mean(
                                [e.sum() for e in a.individual_variance_explained_]
                            ),
                            a.noise_variance_explained_.mean(),
                        ]
                    )
            o = np.mean(ours, axis=0)
            print(
                f"{'':<9}{rate:>14.0%}{'JIVEEY':>9}{o[0]:>8.2f}{o[1]:>12.2f}{o[2]:>8.2f}"
            )
            if theirs:
                t = np.mean(theirs, axis=0)
                print(
                    f"{'':<9}{rate:>14.0%}{'AJIVE':>9}{t[0]:>8.2f}{t[1]:>12.2f}{t[2]:>8.2f}"
                )


if __name__ == "__main__":
    main()
