"""Replicate a MOFA analysis of the CLL cohort with the missing-view CCA models.

Two questions, with MOFA+ as the comparison:

1. Do the shared factors, fitted without them, recover IGHV status and trisomy 12?
   The cross-validated AUC of a logistic regression on each method's 10 factors.
2. Can a method fill in a view that was hidden for some patients? Skill over
   predicting the view's mean, for the drug and mRNA views, on patients whose
   view was hidden at random and who have every other view.

    python run.py --out output/ [--workers 4]
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import data
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent / "missing_views"))
import methods

from cca_zoo.linear import IncompleteGCCA

N_FACTORS = 10
HIDE_FRACTION = 0.25
HIDDEN_VIEWS = ("Drugs", "mRNA")
N_REPEATS = 5
METHODS = (
    "IncompleteGCCA",
    "Complete cases",
    "Mean-imputed",
    "MOFA+ (default)",
    "MOFA+ (best of 5 random)",
)
Views = list[np.ndarray]


def _incomplete(views: Views, seed: int) -> tuple[IncompleteGCCA, np.ndarray]:
    """IncompleteGCCA with its shrinkage tuned; its shared latent."""
    model = methods._tuned(
        lambda s: IncompleteGCCA(n_components=N_FACTORS, shrinkage=s), views, seed
    )
    return model, model.transform_shared(views)


def _predict_with(model: IncompleteGCCA) -> Any:
    """Predict a view for some patients from their other views, as for new patients."""

    def predict(views: Views, rows: np.ndarray, hidden: int) -> np.ndarray:
        given = [None if j == hidden else v[rows] for j, v in enumerate(views)]
        prediction: np.ndarray = model.predict(given)[hidden]
        return prediction

    return predict


def fit_all(views: Views, seed: int) -> dict[str, Any]:
    """Every method fitted to the views: its latent for every sample, and a predictor.

    The predictor maps (the views, the patients, the view to fill in) to that view's
    prediction for those patients.
    """
    observed = np.stack([~np.isnan(v).all(axis=1) for v in views], axis=1)
    fitted: dict[str, Any] = {}

    model, latent = _incomplete(views, seed)
    # The patients to fill in are part of the fit, which is what impute is for.
    fitted["IncompleteGCCA"] = (latent, lambda v, r, h, m=model: m.impute(v)[h][r])

    complete = observed.all(axis=1)
    model, _ = _incomplete([v[complete] for v in views], seed)
    fitted["Complete cases"] = (model.transform_shared(views), _predict_with(model))

    filled = [np.nan_to_num(v) for v in views]
    model, latent = _incomplete(filled, seed)
    fitted["Mean-imputed"] = (latent, _predict_with(model))

    for name, n_init in (("MOFA+ (default)", 1), ("MOFA+ (best of 5 random)", 5)):
        fits = [
            methods._mofa_once(views, N_FACTORS, 1000 * seed + i, n_init > 1)
            for i in range(n_init)
        ]
        z, weights, means, _ = max(fits, key=lambda f: f[3])
        fitted[name] = (z, (z, weights, means))
    return fitted


def predict_hidden(
    predictor: Any, views: Views, rows: np.ndarray, hidden: int
) -> np.ndarray:
    """A method's prediction of view ``hidden`` for the patients ``rows``."""
    if isinstance(predictor, tuple):  # MOFA+: z W' + mean
        z, weights, means = predictor
        return z[rows] @ weights[hidden].T + means[hidden]
    prediction: np.ndarray = predictor(views, rows, hidden)
    return prediction


def biology(cll: data.CLL, seed: int = 0) -> list[dict]:
    """Cross-validated AUC of IGHV and trisomy 12 from each method's factors."""
    fitted = fit_all(cll.views, seed)
    rows, latents = [], {}
    for name, (latent, _) in fitted.items():
        latents[name] = latent.tolist()
        for label, y in cll.labels.items():
            known = ~np.isnan(y)
            splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
            aucs = []
            for train, test in splitter.split(latent[known], y[known]):
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0))
                clf.fit(latent[known][train], y[known][train])
                score = clf.predict_proba(latent[known][test])[:, 1]
                aucs.append(roc_auc_score(y[known][test], score))
            single = [
                max(
                    roc_auc_score(y[known], latent[known][:, k]),
                    1 - roc_auc_score(y[known], latent[known][:, k]),
                )
                for k in range(latent.shape[1])
            ]
            rows.append(
                {
                    "task": "biology",
                    "method": name,
                    "label": label,
                    "auc": float(np.mean(aucs)),
                    "auc_sd": float(np.std(aucs)),
                    "best_single_factor_auc": float(max(single)),
                }
            )
    rows.append(
        {"task": "latents", "latents": latents, "IGHV": cll.labels["IGHV"].tolist()}
    )
    return rows


def hold_out(cll: data.CLL, view_name: str, repeat: int) -> list[dict]:
    """Hide a view for a quarter of the patients with all others; fill it in."""
    rng = np.random.default_rng(repeat)
    hidden = cll.names.index(view_name)
    candidates = np.flatnonzero(cll.observed.all(axis=1))
    rows = np.sort(
        rng.choice(candidates, int(HIDE_FRACTION * len(candidates)), replace=False)
    )
    truth = cll.views[hidden][rows]

    views = [v.copy() for v in cll.views]
    # Standardise the hidden view on the patients that keep it.
    keep = cll.observed[:, hidden].copy()
    keep[rows] = False
    mean, sd = views[hidden][keep].mean(axis=0), views[hidden][keep].std(axis=0)
    views[hidden] = (views[hidden] - mean) / np.where(sd > 0, sd, 1.0)
    truth = views[hidden][rows].copy()
    views[hidden][rows] = np.nan

    fitted = fit_all(views, repeat)
    baseline = np.nanmean(views[hidden], axis=0)
    results = []
    for name, (_, predictor) in fitted.items():
        predicted = predict_hidden(predictor, views, rows, hidden)
        skill = 1 - np.sum((predicted - truth) ** 2) / np.sum((baseline - truth) ** 2)
        results.append(
            {
                "task": "hold_out",
                "method": name,
                "view": view_name,
                "repeat": repeat,
                "skill": float(skill),
            }
        )
    return results


def _task(task: tuple, cll: data.CLL) -> list[dict]:
    kind, *args = task
    return biology(cll) if kind == "biology" else hold_out(cll, *args)


def main() -> None:
    """Run every task and write the results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    cll = data.load()
    tasks = [("biology",)] + [
        ("hold_out", view, repeat)
        for view in HIDDEN_VIEWS
        for repeat in range(N_REPEATS)
    ]
    with Pool(args.workers) as pool:
        rows = []
        for done, result in enumerate(
            pool.imap_unordered(partial(_task, cll=cll), tasks), 1
        ):
            rows.extend(result)
            print(f"{done}/{len(tasks)}", flush=True)
    args.out.mkdir(exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
