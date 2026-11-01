"""E3: the CLL cohort. Joint and individual structure of drugs, methylation, mRNA, mutations.

python e3_cll.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
from common import (
    HERE,
    JIVEEY,
    ajive_fit,
    mean_filled,
    mofa_joint_individual,
    select_ranks,
    select_shrinkage,
)
from e2_variance import cross_fitted_shares
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(HERE.parent / "cll"))
import data  # noqa: E402


def auc(features: np.ndarray, y: np.ndarray) -> float:
    """Cross-validated AUC of a logistic regression of the label on the features."""
    known = ~np.isnan(y)
    aucs = []
    for train, test in RepeatedStratifiedKFold(
        n_splits=5, n_repeats=5, random_state=0
    ).split(features[known], y[known]):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0))
        clf.fit(features[known][train], y[known][train])
        aucs.append(
            roc_auc_score(y[known][test], clf.predict_proba(features[known][test])[:, 1])
        )
    return float(np.mean(aucs))


def main() -> None:
    """Fit, then report ranks, variance shares and what the scores say about IGHV and trisomy 12."""
    cll = data.load()
    views = [np.where(o[:, None], v, np.nan) for v, o in zip(cll.views, cll.observed.T)]
    out: dict = {"names": cll.names, "observed_fraction": cll.observed.mean(0).tolist()}

    k, r = select_ranks(
        views, 0.5, joint=(1, 2, 3, 4, 5, 6), individual=(0, 1, 2, 3, 4), seed=0
    )
    c = select_shrinkage(views, k, r)
    out.update(k=k, r=r, c=c)
    print("selected", k, r, c, flush=True)

    model = JIVEEY(k, r, c).fit(views, seed=0)
    joint = model.transform_joint(views)
    individual = model.transform_individual(views)
    out["jiveey_shares_in_sample"] = {
        key: np.asarray(v).tolist() for key, v in model.variance_explained(views).items()
    }
    out["jiveey_shares_cross_fitted"] = cross_fitted_shares(views, k, r, c, 0).tolist()

    out["auc"] = {}
    for label, y in cll.labels.items():
        row = {"JIVEEY joint": auc(joint, y)}
        for name, scores in zip(cll.names, individual):
            row[f"JIVEEY individual, {name}"] = auc(np.nan_to_num(scores), y)
        out["auc"][label] = row
    # Which single component carries each label?
    out["best_component"] = {}
    for label, y in cll.labels.items():
        known = ~np.isnan(y)
        a = [
            max(roc_auc_score(y[known], joint[known, j]), 1 - roc_auc_score(y[known], joint[known, j]))
            for j in range(k)
        ]
        out["best_component"][label] = {"joint": [float(x) for x in a]}
        for name, scores in zip(cll.names, individual):
            s = np.nan_to_num(scores)
            out["best_component"][label][name] = [
                float(max(roc_auc_score(y[known], s[known, j]), 1 - roc_auc_score(y[known], s[known, j])))
                for j in range(s.shape[1])
            ]

    a = ajive_fit(views, cll.observed)
    out["ajive"] = dict(
        k=int(a.joint_scores_.shape[1]),
        joint_share=np.asarray(a.joint_variance_explained_).sum(-1).tolist()
        if np.asarray(a.joint_variance_explained_).ndim > 1
        else None,
    )
    ajive_joint = np.mean(a.transform(mean_filled(views, cll.observed)), axis=0)
    out["auc_ajive_joint"] = {label: auc(ajive_joint, y) for label, y in cll.labels.items()}

    s, p = mofa_joint_individual(views, cll.observed, k, r, 0)
    out["auc_mofa_shared"] = {label: auc(s, y) for label, y in cll.labels.items()}
    out["auc_mofa_private"] = {
        name: {label: auc(z, y) for label, y in cll.labels.items()}
        for name, z in zip(cll.names, p)
    }
    (HERE / "output" / "e3.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["auc"], indent=1))


if __name__ == "__main__":
    main()
