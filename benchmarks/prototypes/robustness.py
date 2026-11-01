"""How the ridge-whitened EY shared/private fit copes with the wrong ranks.

The simulator has two shared factors and three private factors per view. Each method is
given a shared rank k and a private rank r, right and wrong, and scored on recovery of the
true shared and private factors (mean squared cosine of the principal angles, over as many
angles as the smaller of the two spaces has).

    python robustness.py [--seeds 8]
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
from ey_shared_private import WhitenedSharedPrivateEY

sys.path.insert(0, str(Path(__file__).parent.parent / "missing_views"))
from metrics import subspace_recovery  # noqa: E402
from simulate import simulate  # noqa: E402

from cca_zoo.linear import AJIVE  # noqa: E402

DESIGN = dict(
    n_samples=150, widths=(300, 400, 200), n_shared=2, n_private=3, private=0.8
)


def private_recovery(
    scores: list[np.ndarray], truth: list[np.ndarray], observed: np.ndarray
) -> float:
    """Mean over views of the private-factor recovery on the samples that have the view."""
    return float(
        np.mean(
            [
                subspace_recovery(s[o], t[o])
                for s, t, o in zip(scores, truth, observed.T)
            ]
        )
    )


def main() -> None:
    """Print recovery for right and wrong ranks, complete and with views missing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=8)
    args = parser.parse_args()
    import pandas as pd

    rows = []
    for shared, rate, k, r, seed in itertools.product(
        (0.02,), (0.0, 0.3), (1, 2, 3), (1, 3, 5), range(args.seeds)
    ):
        ds = simulate(seed, shared=shared, missing_rate=rate, **DESIGN)
        masked = ds.masked
        fit = WhitenedSharedPrivateEY(k, r, shrinkage=0.5).fit(masked, seed=seed)
        latent, private = fit.scores(masked)
        rows.append(
            dict(
                method="EY whitened",
                rate=rate,
                k=k,
                r=r,
                shared=subspace_recovery(latent, ds.shared_factors),
                private=private_recovery(private, ds.private_factors, ds.observed),
            )
        )
        if rate == 0.0:
            ajive = AJIVE(n_components=k, individual_ranks=r).fit(ds.views)
            rows.append(
                dict(
                    method="AJIVE",
                    rate=rate,
                    k=k,
                    r=r,
                    shared=subspace_recovery(
                        np.mean(ajive.transform(ds.views), axis=0), ds.shared_factors
                    ),
                    private=private_recovery(
                        ajive.individual_scores_, ds.private_factors, ds.observed
                    ),
                )
            )
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(
        df.groupby(["rate", "method", "k", "r"])[["shared", "private"]]
        .mean()
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
