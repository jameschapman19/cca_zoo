"""Shared pieces of the JIVEEY paper experiments: model selection and the baselines."""

from __future__ import annotations

import copy
import sys
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "prototypes"))
sys.path.insert(0, str(HERE.parent / "missing_views"))
from jiveey import JIVEEY  # noqa: E402
from methods import _mofa_once, _shared_factors  # noqa: E402
from metrics import subspace_recovery  # noqa: E402

from cca_zoo.linear import AJIVE, IncompleteGCCA  # noqa: E402
from cca_zoo.metrics import variance_explained  # noqa: E402

Views = list[np.ndarray]

REGIMES = {
    "moderate": dict(
        n_samples=150, widths=(60, 80, 40), n_private=2, private=0.5, shared=0.1
    ),
    "faint": dict(
        n_samples=150, widths=(300, 400, 200), n_private=3, private=0.8, shared=0.02
    ),
    "fainter": dict(
        n_samples=150, widths=(300, 400, 200), n_private=3, private=0.8, shared=0.01
    ),
}
N_SHARED = 2


def heldout_loss(model: JIVEEY, views: Views) -> float:
    """The model's own loss, with its weights fixed, evaluated on other samples."""
    other = copy.copy(model)
    other._prepare(views)
    return other.loss_and_gradient(other._flatten(model.weights_))[0]


def heldout_correlation(model: JIVEEY, views: Views) -> float:
    """Mean pairwise correlation of the views' joint scores over held-out samples."""
    seen = np.stack([~np.isnan(v).any(axis=1) for v in views], axis=1)
    scores = [z[:, : model.n_joint] for z in model._projections(views)]
    out = []
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            both = seen[:, i] & seen[:, j]
            a, b = scores[i][both], scores[j][both]
            out.append(
                np.mean(
                    [np.corrcoef(a[:, k], b[:, k])[0, 1] for k in range(a.shape[1])]
                )
            )
    return float(np.mean(out))


def _folds(views: Views, seed: int, n_splits: int = 3):
    return KFold(n_splits, shuffle=True, random_state=seed).split(views[0])


def select_ranks(
    views: Views,
    shrinkage: float = 0.5,
    joint: tuple[int, ...] = (1, 2, 3, 4),
    individual: tuple[int, ...] = (0, 1, 2, 3, 4, 5),
    seed: int = 0,
) -> tuple[int, int]:
    """The joint rank and the (common) individual rank with the lowest held-out loss."""
    best, arg = np.inf, (joint[0], individual[0])
    for k, r in product(joint, individual):
        losses = []
        for train, test in _folds(views, seed):
            model = JIVEEY(k, r, shrinkage).fit(
                [v[train] for v in views], seed=seed, n_init=1
            )
            losses.append(heldout_loss(model, [v[test] for v in views]))
        if np.mean(losses) < best:
            best, arg = float(np.mean(losses)), (k, r)
    return arg


def select_shrinkage(
    views: Views,
    k: int,
    r: int,
    grid: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9),
    seed: int = 0,
) -> float:
    """The shrinkage with the highest held-out correlation of the joint scores."""
    scores = []
    for c in grid:
        fold = []
        for train, test in _folds(views, seed):
            model = JIVEEY(k, r, c).fit([v[train] for v in views], seed=seed, n_init=1)
            fold.append(heldout_correlation(model, [v[test] for v in views]))
        scores.append(np.mean(fold))
    return grid[int(np.argmax(scores))]


def private_recovery(scores, truth, observed) -> float:
    """Mean over views of the individual-factor recovery on the samples that have the view."""
    return float(
        np.mean(
            [subspace_recovery(s[o], t[o]) for s, t, o in zip(scores, truth, observed.T)]
        )
    )


def mean_filled(views: Views, observed: np.ndarray) -> Views:
    """Views with the rows of samples that lack them set to the column mean."""
    return [
        np.where(o[:, None], v, np.nanmean(v, axis=0)) for v, o in zip(views, observed.T)
    ]


def ajive_fit(views: Views, observed: np.ndarray, k=None, r=None) -> AJIVE:
    """AJIVE on mean-filled views, with given or automatically selected ranks."""
    filled = mean_filled(views, observed)
    model = AJIVE(
        n_components="auto" if k is None else k,
        individual_ranks=r,
    )
    return model.fit(filled)


def mofa_joint_individual(
    views: Views, observed: np.ndarray, k: int, r: int, seed: int
) -> tuple[np.ndarray, list[np.ndarray]]:
    """What a MOFA+ user does: shared = factors active in every view, private = the rest by view.

    The model gets the true total number of factors. Each remaining factor is assigned to the
    view in which it explains the most variance.
    """
    m = len(views)
    z, _, _, _ = _mofa_once(views, k + m * r, seed, False)
    shared = _shared_factors(z, views, k)
    centred = [v - np.nanmean(v, axis=0) for v in views]
    r2 = variance_explained(centred, [z] * m)  # (views, factors)
    shared_idx = [int(np.argmin(np.abs(z - s[:, None]).sum(0))) for s in shared.T]
    owner = np.argmax(r2, axis=0)
    private = []
    for i in range(m):
        idx = [f for f in range(z.shape[1]) if owner[f] == i and f not in shared_idx]
        private.append(z[:, idx] if idx else np.zeros((len(z), 1)))
    return shared, private
