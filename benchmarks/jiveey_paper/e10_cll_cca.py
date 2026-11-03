"""E10: CLL labels from automatic PCA-reduced GCCA, AJIVE (mean-filled and EM) and MOFA+.

On the cohort as it is, and with 30% of the patients who have several views losing one at random
(5 repeats). Only MOFA+ is told the ranks: the joint rank and individual ranks PCA-reduced GCCA chose.

python e10_cll_cca.py
"""

from __future__ import annotations

import json
import sys
from multiprocessing import Pool

import numpy as np
from ajive_em import ajive_em_final
from common import HERE, ajive_fit, mean_filled, mofa_joint_individual
from e3_cll import auc
from pca_gcca import fit_pca_gcca

sys.path.insert(0, str(HERE.parent / "cll"))
import data  # noqa: E402


def run(repeat: int) -> list[dict]:
    """All methods on the cohort, with extra views removed unless ``repeat`` is negative."""
    cll = data.load()
    observed = cll.observed.copy()
    if repeat >= 0:
        rng = np.random.default_rng(repeat)
        eligible = np.flatnonzero(observed.sum(1) >= 2)
        for i in rng.choice(eligible, int(0.3 * len(eligible)), replace=False):
            observed[i, rng.choice(np.flatnonzero(observed[i]))] = False
    views = [np.where(o[:, None], v, np.nan) for v, o in zip(cll.views, observed.T)]
    scores, ranks = {}, {}
    fit = fit_pca_gcca(views, observed, 0)
    scores["PCA-reduced GCCA"] = fit[0]
    ranks["PCA-reduced GCCA"] = fit[2]
    fit_em = ajive_em_final(views, observed)
    scores["AJIVE-EM"] = fit_em[0]
    try:
        a = ajive_fit(views, observed)
        scores["AJIVE, mean-filled"] = np.mean(a.transform(mean_filled(views, observed)), axis=0)
    except ValueError:
        pass
    k, signal = fit[2]["joint"], fit[2]["signal"]
    r = max(int(np.round(np.mean([s - k for s in signal]))), 1)
    scores["MOFA+"] = mofa_joint_individual(views, observed, k, r, max(repeat, 0))[0]
    return [
        dict(repeat=repeat, method=m, label=label, auc=auc(z, y), joint_rank=int(z.shape[1]), signal=ranks.get(m, {}).get("signal"))
        for m, z in scores.items()
        for label, y in cll.labels.items()
    ]


if __name__ == "__main__":
    with Pool(2) as pool:
        rows = [r for out in pool.map(run, [-1, 0, 1, 2, 3, 4]) for r in out]
    (HERE / "output" / "e10.json").write_text(json.dumps(rows))
