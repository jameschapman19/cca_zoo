"""E6: CLL with extra views removed. How much of IGHV and trisomy 12 do the joint scores keep?

For each repeat, 30% of the patients who have at least two views lose one of them at random. Each
method is fitted on what remains, at fixed ranks (joint 5, individual 4, shrinkage 0.9, as selected on
the untouched data), and a logistic regression on the joint scores of all patients is cross-validated.

python e6_cll_missing.py
"""

from __future__ import annotations

import json
import sys
from multiprocessing import Pool

import numpy as np
from common import HERE, JIVEEY, ajive_fit, em_complete, mean_filled, mofa_joint_individual
from e3_cll import auc

sys.path.insert(0, str(HERE.parent / "cll"))
import data  # noqa: E402

K, R, C, FRACTION = 5, 4, 0.9, 0.3


def run(repeat: int) -> list[dict]:
    """All methods on the cohort with extra views removed."""
    cll = data.load()
    rng = np.random.default_rng(repeat)
    observed = cll.observed.copy()
    eligible = np.flatnonzero(observed.sum(1) >= 2)
    for i in rng.choice(eligible, int(FRACTION * len(eligible)), replace=False):
        observed[i, rng.choice(np.flatnonzero(observed[i]))] = False
    views = [np.where(o[:, None], v, np.nan) for v, o in zip(cll.views, observed.T)]
    scores = {}
    scores["JIVEEY"] = JIVEEY(K, R, C).fit(views, seed=repeat).transform_joint(views)
    a = ajive_fit(views, observed, K, R)
    scores["AJIVE, mean-filled"] = np.mean(a.transform(mean_filled(views, observed)), axis=0)
    done = em_complete(views, observed, K + len(views) * R)
    a = ajive_fit(done, np.ones_like(observed), K, R)
    scores["AJIVE, EM-completed"] = np.mean(a.transform(done), axis=0)
    scores["MOFA+"] = mofa_joint_individual(views, observed, K, R, repeat)[0]
    return [
        dict(repeat=repeat, method=m, label=label, auc=auc(z, y))
        for m, z in scores.items()
        for label, y in cll.labels.items()
    ]


if __name__ == "__main__":
    with Pool(2) as pool:
        rows = [r for out in pool.map(run, range(5)) for r in out]
    (HERE / "output" / "e6.json").write_text(json.dumps(rows))
