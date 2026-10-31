"""The methods compared, each fitted to views with whole samples missing.

Every method returns ``(latent, reconstruction)``: an array of shape
(n_samples, n_shared) with the shared factors it estimates, and a list with
each view reconstructed for every sample from the views the sample has.
"""

from __future__ import annotations

import contextlib
import io
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from mofapy2.run.entry_point import entry_point
from sklearn.model_selection import KFold

from cca_zoo.linear import CCAEY, GCCA, IncompleteGCCA
from cca_zoo.metrics import variance_explained
from cca_zoo.probabilistic import GFA

SHRINKAGES = (0.05, 0.2, 0.5, 0.8)

Views = list[np.ndarray]
Result = tuple[np.ndarray, Views]


def _tuned(make: Callable[[float], Any], views: Views, seed: int) -> Any:
    """``make(shrinkage)`` fitted to all samples, shrinkage chosen by 3-fold CV.

    Folds split samples; each model is scored on the held-out samples by the
    mean pairwise correlation of its scores.
    """
    scores = []
    for shrinkage in SHRINKAGES:
        fold_scores = []
        for train, test in KFold(3, shuffle=True, random_state=seed).split(views[0]):
            model = make(shrinkage).fit([v[train] for v in views])
            fold_scores.append(model.score([v[test] for v in views]))
        scores.append(np.mean(fold_scores))
    return make(SHRINKAGES[int(np.argmax(scores))]).fit(views)


def _reconstruct(
    model: object, views: list[np.ndarray], observed: np.ndarray
) -> list[np.ndarray]:
    """Each view for every sample, from the views that sample has.

    Samples are grouped by which views they have, since ``predict`` takes the
    same views for all its samples.
    """
    n = len(views[0])
    out = [np.empty_like(v) for v in views]
    for pattern in np.unique(observed, axis=0):
        rows = np.flatnonzero((observed == pattern).all(axis=1))
        given = [v[rows] if seen else None for v, seen in zip(views, pattern)]
        for target, predicted in zip(out, model.predict(given)):
            target[rows] = predicted
    assert out[0].shape[0] == n
    return out


def _shared_factors(
    latent: np.ndarray, views: list[np.ndarray], n_shared: int
) -> np.ndarray:
    """The ``n_shared`` factors that explain the most variance in every view.

    What a practitioner does with MOFA's output: keep the factors active in all
    views, ranked by the least variance any view has in them.
    """
    centred = [v - np.nanmean(v, axis=0) for v in views]
    least = variance_explained(centred, [latent] * len(views)).min(axis=0)
    return latent[:, np.argsort(least)[::-1][:n_shared]]


def incomplete_gcca(
    views: Views,
    observed: np.ndarray,
    n_shared: int,
    seed: int,
) -> Result:
    model = _tuned(
        lambda s: IncompleteGCCA(n_components=n_shared, shrinkage=s), views, seed
    )
    return model.transform_shared(views), _reconstruct(model, views, observed)


def masked_ccaey(
    views: Views,
    observed: np.ndarray,
    n_shared: int,
    seed: int,
) -> Result:
    model = _tuned(
        lambda s: CCAEY(n_components=n_shared, shrinkage=s, random_state=seed),
        views,
        seed,
    )
    return model.transform_shared(views), _reconstruct(model, views, observed)


def gcca_complete_cases(
    views: Views,
    observed: np.ndarray,
    n_shared: int,
    seed: int,
) -> Result:
    """GCCA on the samples that have every view; the rest are projected."""
    complete = observed.all(axis=1)
    model = _tuned(
        lambda s: GCCA(n_components=n_shared, shrinkage=s),
        [v[complete] for v in views],
        seed,
    )
    return _mean_observed_scores(model, views, observed), _reconstruct(
        model, views, observed
    )


def gcca_zero_filled(
    views: Views,
    observed: np.ndarray,
    n_shared: int,
    seed: int,
) -> Result:
    """GCCA with each missing view replaced by its mean (zero once centred)."""
    means = [np.nanmean(v, axis=0) for v in views]
    centred = [v - m for v, m in zip(views, means)]
    model = _tuned(
        lambda s: GCCA(n_components=n_shared, shrinkage=s),
        [np.nan_to_num(v) for v in centred],
        seed,
    )
    latent = _mean_observed_scores(model, centred, observed)
    return latent, [
        r + m for r, m in zip(_reconstruct(model, centred, observed), means)
    ]


def _mean_observed_scores(model: Any, views: Views, observed: np.ndarray) -> np.ndarray:
    """Each sample's scores averaged over the views it has."""
    scores = []
    for i, v in enumerate(views):
        given = [None] * len(views)
        given[i] = np.nan_to_num(v)
        scores.append(model.transform_shared(given))
    seen = observed.T[:, :, None].astype(float)
    return (np.stack(scores) * seen).sum(axis=0) / seen.sum(axis=0)


def gfa_zero_filled(
    views: Views,
    observed: np.ndarray,
    n_shared: int,
    seed: int,
) -> Result:
    """GFA fitted to zero-filled views, which it has no way to mask here.

    Its reconstruction is still the posterior given each sample's own views.
    """
    means = [np.nanmean(v, axis=0) for v in views]
    centred = [v - m for v, m in zip(views, means)]
    model = GFA(
        n_components=n_shared + 3 * 2,
        max_iter=2000,
        n_posterior_samples=10,
        random_state=seed,
    ).fit([np.nan_to_num(v) for v in centred])
    latent = model.transform_shared([np.nan_to_num(v) for v in centred])
    return _shared_factors(latent, views, n_shared), [
        r + m for r, m in zip(_reconstruct(model, centred, observed), means)
    ]


def _mofa_once(
    views: Views, n_factors: int, seed: int
) -> tuple[np.ndarray, Views, Views, float]:
    """One MOFA+ fit on the NaN-masked views: factors, weights, means, final ELBO."""
    means = [np.nanmean(v, axis=0) for v in views]
    with contextlib.redirect_stdout(io.StringIO()):  # MOFA prints its whole log
        run = entry_point()
        run.set_data_options(scale_views=False)
        run.set_data_matrix(
            [[v - m] for v, m in zip(views, means)],
            likelihoods=["gaussian"] * len(views),
            views_names=[f"v{i}" for i in range(len(views))],
            groups_names=["g"],
            samples_names=[[f"s{j}" for j in range(len(views[0]))]],
            features_names=[
                [f"v{i}_{j}" for j in range(v.shape[1])] for i, v in enumerate(views)
            ],
        )
        run.set_model_options(
            factors=n_factors, spikeslab_weights=True, ard_weights=True
        )
        # dropR2=None keeps every factor, so that the factors compared are the ones
        # the model was given rather than the ones it chose to keep.
        run.set_train_options(
            iter=1000,
            convergence_mode="fast",
            dropR2=None,
            seed=seed,
            verbose=False,
            quiet=True,
        )
        run.build()
        run.run()
    elbo = np.asarray(run.model.train_stats["elbo"])
    return (
        run.model.nodes["Z"].getExpectation(),
        run.model.nodes["W"].getExpectation(),
        means,
        elbo[np.isfinite(elbo)][-1],
    )


def _mofa(
    views: Views, observed: np.ndarray, n_shared: int, seed: int, n_init: int
) -> Result:
    """MOFA+ with as many factors as were simulated; the best of ``n_init`` fits.

    The fit with the highest ELBO is kept, as the MOFA documentation advises,
    since variational inference ends in different local optima.
    """
    fits = [
        _mofa_once(views, n_shared + 3 * 2, 1000 * seed + init)
        for init in range(n_init)
    ]
    z, weights, means, _ = max(fits, key=lambda fit: fit[3])
    latent = _shared_factors(z, views, n_shared)
    return latent, [z @ w.T + m for w, m in zip(weights, means)]


def mofa_single_init(
    views: Views,
    observed: np.ndarray,
    n_shared: int,
    seed: int,
) -> Result:
    return _mofa(views, observed, n_shared, seed, n_init=1)


def mofa_best_of_five(
    views: Views,
    observed: np.ndarray,
    n_shared: int,
    seed: int,
) -> Result:
    return _mofa(views, observed, n_shared, seed, n_init=5)


METHODS: dict[str, Callable[..., Result]] = {
    "IncompleteGCCA": incomplete_gcca,
    "CCAEY (masked)": masked_ccaey,
    "GCCA, complete cases": gcca_complete_cases,
    "GCCA, zero-filled": gcca_zero_filled,
    "GFA, zero-filled": gfa_zero_filled,
    "MOFA+ (1 init)": mofa_single_init,
    "MOFA+ (best of 5)": mofa_best_of_five,
}


def timed(method: Callable[..., Result], *args: Any) -> tuple[Result, float]:
    """The method's result and its wall-clock seconds."""
    start = time.perf_counter()
    result = method(*args)
    return result, time.perf_counter() - start
