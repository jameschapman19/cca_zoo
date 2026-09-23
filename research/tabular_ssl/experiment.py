"""Label-scarce evaluation of tabular SSL representations.

For one table: split into an unlabeled pool (75%) and a test set (25%). Every
SSL method is fit on the pool *without labels*. Then, for each label budget,
``reps`` random labelled subsets are drawn from the pool and two probes -- a
cross-validated linear model and XGBoost with fixed small-data settings -- are
trained on each feature set and scored on the test set (R^2 for regression,
accuracy for classification).

The number of components kept by the spectral methods is set without labels:
components whose view agreement exceeds ``1 - mask_rate`` (the agreement of a
purely idiosyncratic column is ``(1 - mask_rate)^2``; ``1 - mask_rate`` is the
geometric midpoint between that floor and perfect agreement). PCA gets the
same number of components.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from tabssl import ConditionalSSL, CopulaTransform, SpectralSSL

MASK_RATE = 0.5
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


def representations(Zp: np.ndarray, Zt: np.ndarray, seed: int) -> dict:
    """Fit every SSL method on pool ``Zp``; return (pool, test) feature pairs."""
    gibbs = SpectralSSL("gibbs", MASK_RATE, random_state=seed).fit(Zp)
    marginal = SpectralSSL("marginal", MASK_RATE, random_state=seed).fit(Zp)
    k = max(1, int(np.sum(gibbs.agreement_ > 1 - MASK_RATE)))
    pca = PCA(k, random_state=seed).fit(Zp)
    cond = ConditionalSSL(random_state=seed)
    Cp = cond.fit_transform(Zp)
    Ct = cond.transform(Zt)
    reps = {
        "raw": (Zp, Zt),
        "pca": (pca.transform(Zp), pca.transform(Zt)),
        "spectral-marginal": (marginal.transform(Zp, k), marginal.transform(Zt, k)),
        "spectral-gibbs": (gibbs.transform(Zp, k), gibbs.transform(Zt, k)),
        "conditional": (Cp, Ct),
    }
    for name in ["pca", "spectral-gibbs", "conditional"]:
        p, t = reps[name]
        reps[f"raw+{name}"] = (np.hstack([Zp, p]), np.hstack([Zt, t]))
    return reps, k


def _probe(kind: str, task: str, seed: int):  # noqa: ANN202
    if kind == "linear":
        if task == "reg":
            return RidgeCV(alphas=np.logspace(-3, 3, 13))
        return LogisticRegressionCV(Cs=np.logspace(-3, 3, 7), cv=3, max_iter=2000)
    params = dict(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=1,
        random_state=seed,
    )
    return XGBRegressor(**params) if task == "reg" else XGBClassifier(**params)


def evaluate(
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    budgets: list[int],
    reps: int,
    seed: int,
    dataset: str,
) -> list[dict]:
    """Run the full protocol on one pool/test split of one table."""
    if task != "reg":
        y = np.unique(y, return_inverse=True)[1]
    strat = None if task == "reg" else y
    Xp, Xt, yp, yt = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=strat
    )
    varying = np.ptp(Xp, axis=0) > 0
    Xp, Xt = Xp[:, varying], Xt[:, varying]
    cop = CopulaTransform().fit(Xp)
    feats, k = representations(cop.transform(Xp), cop.transform(Xt), seed)
    rng = np.random.default_rng(seed)
    rows = []
    for b in budgets:
        if b > len(yp) // 2:
            continue
        for r in range(reps):
            idx = _draw(yp, b, task, rng)
            for name, (Fp, Ft) in feats.items():
                for kind in ["linear", "xgb"]:
                    if kind == "linear" and name.startswith("raw+"):
                        continue
                    sc = StandardScaler().fit(Fp[idx])
                    model = _probe(kind, task, seed + r)
                    model.fit(sc.transform(Fp[idx]), yp[idx])
                    pred = model.predict(sc.transform(Ft))
                    score = r2_score(yt, pred) if task == "reg" else accuracy_score(yt, pred)
                    rows.append(
                        dict(dataset=dataset, split=seed, budget=b, rep=r,
                             features=name, probe=kind, score=score, k=k)
                    )
    return rows


def _draw(y: np.ndarray, b: int, task: str, rng: np.random.Generator) -> np.ndarray:
    """Labelled subset of size ``b``; classification keeps >= 3 per class."""
    if task == "reg":
        return rng.choice(len(y), b, replace=False)
    classes = np.unique(y)
    base = np.concatenate([rng.choice(np.flatnonzero(y == c), 3, replace=False)
                           for c in classes])
    rest = np.setdiff1d(np.arange(len(y)), base)
    extra = rng.choice(rest, max(0, b - len(base)), replace=False)
    return np.concatenate([base, extra])
