"""Benchmark CanoBoost against XGBoost on real tabular datasets.

Protocol, identical for every method: 5 random 80/20 train/test splits; the
training part is split again 80/20 into fit/validation. Every configuration in a
method's 4-point grid is trained with early stopping on the validation loss
(max 2000 rounds, learning rate 0.05, row subsample 0.8), the configuration
with the best validation loss is kept, and its test loss is reported.

Methods:
    xgboost        XGBoost (hist), grid max_depth x min_child_weight.
    canoboost      CanonicalBoosting*, grid max_depth x min_samples_leaf.
    axis-aligned   Same implementation, n_components=0 (ablation: no CCA).
    random-proj    Same implementation, directions drawn at random instead of
                   by CCA (ablation: is it the CCA or just any oblique feature?).

Usage:
    python benchmarks/canonical_boosting/run.py [--rotate] [--out results.csv]

``--rotate`` applies a random orthogonal rotation to the (standardised)
features first -- the setting in which axis-aligned trees are known to break.
"""

from __future__ import annotations

import argparse
import io
import itertools
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import ortho_group
from sklearn import datasets as skd
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from cca_zoo.boosting import CanonicalBoostingClassifier, CanonicalBoostingRegressor

CACHE = Path(__file__).parent / ".data"
UCI = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
HOUSING = (
    "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
    "datasets/housing/housing.csv"
)
N_SPLITS = 5
MAX_ROUNDS = 2000
PATIENCE = 50
LR = 0.05


def _csv(url: str, **kw: object) -> pd.DataFrame:
    CACHE.mkdir(exist_ok=True)
    path = CACHE / url.split("/")[3] / url.rsplit("/", 1)[-1]
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        path.write_bytes(urllib.request.urlopen(url).read())
    return pd.read_csv(io.BytesIO(path.read_bytes()), **kw)


def _xy(df: pd.DataFrame, target: str | int) -> tuple[np.ndarray, np.ndarray]:
    y = df.pop(target).to_numpy()
    X = pd.get_dummies(df, dtype=float).to_numpy(dtype=float)
    return X, y


def load(name: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Return ``(X, y, task)`` with task in {reg, bin, multi}."""
    if name == "california":
        df = _csv(HOUSING)
        df["total_bedrooms"] = df["total_bedrooms"].fillna(
            df["total_bedrooms"].median()
        )
        X, y = _xy(df, "median_house_value")
        return X, np.log(y), "reg"
    if name == "abalone":
        return (*_xy(_csv(UCI + "abalone.csv", header=None), 8), "reg")
    if name == "wine-quality":
        return (*_xy(_csv(UCI + "winequality-white.csv", header=None), 11), "reg")
    if name == "boston":
        return (*_xy(_csv(UCI + "housing.csv", header=None), 13), "reg")
    if name == "diabetes":
        return (*skd.load_diabetes(return_X_y=True), "reg")
    if name == "adult":
        df = _csv(UCI + "adult-all.csv", header=None, na_values="?").dropna()
        # 15k-row subsample keeps the exact-split sklearn trees tractable.
        X, y = _xy(df.sample(15000, random_state=0), 14)
        return X, y, "bin"
    if name == "pima":
        return (*_xy(_csv(UCI + "pima-indians-diabetes.csv", header=None), 8), "bin")
    if name == "sonar":
        return (*_xy(_csv(UCI + "sonar.csv", header=None), 60), "bin")
    if name == "ionosphere":
        return (*_xy(_csv(UCI + "ionosphere.csv", header=None), 34), "bin")
    if name == "breast-cancer":
        return (*skd.load_breast_cancer(return_X_y=True), "bin")
    if name == "glass":
        return (*_xy(_csv(UCI + "glass.csv", header=None), 9), "multi")
    if name == "wine":
        return (*skd.load_wine(return_X_y=True), "multi")
    if name == "digits":
        return (*skd.load_digits(return_X_y=True), "multi")
    raise ValueError(name)


DATASETS = [
    "california",
    "abalone",
    "wine-quality",
    "boston",
    "diabetes",
    "adult",
    "pima",
    "sonar",
    "ionosphere",
    "breast-cancer",
    "glass",
    "wine",
    "digits",
]


class _RandomDirections:
    """Mixin replacing the canonical directions with random orthonormal ones."""

    def _directions(
        self, Z: np.ndarray, G: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        r = min(self.n_components, G.shape[1] if G.shape[1] > 1 else 1)
        Q, _ = np.linalg.qr(rng.standard_normal((Z.shape[1], r)))
        return Q


class RandomProjRegressor(_RandomDirections, CanonicalBoostingRegressor):
    """Ablation: random oblique features in place of CCA directions."""


class RandomProjClassifier(_RandomDirections, CanonicalBoostingClassifier):
    """Ablation: random oblique features in place of CCA directions."""


def _loss(task: str, y: np.ndarray, model: object, X: np.ndarray) -> float:
    if task == "reg":
        return float(np.sqrt(mean_squared_error(y, model.predict(X))))
    return float(log_loss(y, model.predict_proba(X), labels=np.unique(y)))


def _fit_ours(
    cls: type,
    task: str,
    Xf: np.ndarray,
    yf: np.ndarray,
    Xv: np.ndarray,
    yv: np.ndarray,
    seed: int,
    **kw: object,
) -> list:
    models = []
    for depth, leaf in itertools.product([3, 6], [5, 20]):
        m = cls(
            n_estimators=MAX_ROUNDS,
            learning_rate=LR,
            max_depth=depth,
            min_samples_leaf=leaf,
            subsample=0.8,
            random_state=seed,
            **kw,
        )
        models.append(m.fit(Xf, yf, eval_set=(Xv, yv), early_stopping_rounds=PATIENCE))
    return models


def _fit_xgb(
    task: str,
    Xf: np.ndarray,
    yf: np.ndarray,
    Xv: np.ndarray,
    yv: np.ndarray,
    seed: int,
) -> list:
    models = []
    cls = XGBRegressor if task == "reg" else XGBClassifier
    for depth, mcw in itertools.product([3, 6], [1, 5]):
        m = cls(
            n_estimators=MAX_ROUNDS,
            learning_rate=LR,
            max_depth=depth,
            min_child_weight=mcw,
            subsample=0.8,
            tree_method="hist",
            early_stopping_rounds=PATIENCE,
            n_jobs=1,
            random_state=seed,
        )
        models.append(m.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False))
    return models


def run_split(name: str, split: int, rotate: bool) -> list[dict]:
    """Train and score every method on one train/test split of one dataset."""
    X, y, task = load(name)
    if task != "reg":
        y = np.unique(y, return_inverse=True)[1]
    stratify = None if task == "reg" else y
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=split, stratify=stratify
    )
    if rotate:
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
        R = ortho_group.rvs(X.shape[1], random_state=split)
        Xtr, Xte = (Xtr - mu) / sd @ R, (Xte - mu) / sd @ R
    Xf, Xv, yf, yv = train_test_split(
        Xtr,
        ytr,
        test_size=0.2,
        random_state=split,
        stratify=None if task == "reg" else ytr,
    )
    reg = task == "reg"
    methods = {
        "xgboost": lambda: _fit_xgb(task, Xf, yf, Xv, yv, split),
        "canoboost": lambda: _fit_ours(
            CanonicalBoostingRegressor if reg else CanonicalBoostingClassifier,
            task,
            Xf,
            yf,
            Xv,
            yv,
            split,
        ),
        "axis-aligned": lambda: _fit_ours(
            CanonicalBoostingRegressor if reg else CanonicalBoostingClassifier,
            task,
            Xf,
            yf,
            Xv,
            yv,
            split,
            n_components=0,
        ),
        "random-proj": lambda: _fit_ours(
            RandomProjRegressor if reg else RandomProjClassifier,
            task,
            Xf,
            yf,
            Xv,
            yv,
            split,
        ),
    }
    rows = []
    for method, fit in methods.items():
        t = time.perf_counter()
        models = fit()
        seconds = time.perf_counter() - t
        best = min(models, key=lambda m: _loss(task, yv, m, Xv))
        rows.append(
            dict(
                dataset=name,
                task=task,
                split=split,
                method=method,
                n=len(X),
                d=X.shape[1],
                rotated=rotate,
                test_loss=_loss(task, yte, best, Xte),
                test_acc=float(np.mean(best.predict(Xte) == yte))
                if not reg
                else np.nan,
                seconds=seconds,
            )
        )
    print(
        name, split, {r["method"]: round(r["test_loss"], 4) for r in rows}, flush=True
    )
    return rows


def main() -> None:
    """Run the full benchmark and write a CSV of per-split results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument("--out", default=None)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    for name in args.datasets:  # download serially before forking workers
        load(name)
    jobs = [
        delayed(run_split)(name, s, args.rotate)
        for name in args.datasets
        for s in range(N_SPLITS)
    ]
    rows = sum(Parallel(n_jobs=args.n_jobs)(jobs), [])
    out = args.out or Path(__file__).parent / (
        "results_rotated.csv" if args.rotate else "results.csv"
    )
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    main()
