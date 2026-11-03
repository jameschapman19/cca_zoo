"""A CCA-like method for views missing from whole samples, with every rank automatic.

1. Each view is reduced to its signal subspace: the top singular directions of its observed rows,
   as many as Gavish and Donoho's optimal hard threshold keeps.
2. IncompleteGCCA (exact multiset CCA on the samples each view has) is fitted to the reduced views.
   The rank cut is its regulariser, so its shrinkage is 0.
3. The number of canonical components is the number whose cross-view correlation beats a permutation
   null (rows of each view shuffled independently).
4. Each view's individual scores come from its observed rows: the joint scores projected out, and
   the top (signal rank - joint rank) directions of the rest.
"""

from __future__ import annotations

import numpy as np
from ajive_em import observed_signal_ranks
from common import IncompleteGCCA, Views


def reduce_views(views: Views, observed: np.ndarray, ranks: list[int]) -> Views:
    """Scores on each view's top ``ranks`` principal directions; NaN rows stay NaN."""
    out = []
    for v, o, s in zip(views, observed.T, ranks):
        x = v[o] - v[o].mean(0)
        vt = np.linalg.svd(x, full_matrices=False)[2][:s]
        z = np.full((len(v), s), np.nan)
        z[o] = x @ vt.T
        out.append(z)
    return out


def _component_correlations(reduced: Views, k: int) -> np.ndarray:
    """Mean over view pairs of the correlation of each canonical component's scores."""
    model = IncompleteGCCA(n_components=k, shrinkage=0.0).fit(reduced)
    scores = model.transform(reduced)
    seen = [~np.isnan(z).any(1) for z in reduced]
    out = []
    for i in range(len(reduced)):
        for j in range(i + 1, len(reduced)):
            both = seen[i] & seen[j]
            if both.sum() < 4:
                continue
            a, b = scores[i][both], scores[j][both]
            out.append([np.corrcoef(a[:, c], b[:, c])[0, 1] for c in range(k)])
    return np.mean(out, axis=0) if out else np.zeros(k)


def select_joint_rank(reduced: Views, k_max: int = 6, n_perm: int = 30, level: float = 0.95, seed: int = 0) -> int:
    """Leading components whose cross-view correlation beats the permutation null."""
    rng = np.random.default_rng(seed)
    k_max = min(k_max, min(z.shape[1] for z in reduced))
    observed = _component_correlations(reduced, k_max)
    null = np.array(
        [
            _component_correlations([z[rng.permutation(len(z))] for z in reduced], k_max)
            for _ in range(n_perm)
        ]
    )
    above = observed > np.quantile(null, level, axis=0)
    k = 0
    while k < k_max and above[k]:
        k += 1
    return k


def joint_rank_by_ajive(views: Views, observed: np.ndarray, signal: list[int]) -> int:
    """AJIVE's random-direction joint rank on EM-completed views (0 when it finds none)."""
    from common import AJIVE, em_complete

    done = em_complete(views, observed, sum(signal))
    try:
        return int(AJIVE(n_components="auto", signal_ranks=signal).fit(done).joint_scores_.shape[1])
    except ValueError:
        return 0


def fit_pca_gcca(views: Views, observed: np.ndarray, seed: int = 0, rank_rule: str = "native"):
    """The whole pipeline. Returns joint scores, individual scores and the ranks, or None if no joint rank."""
    signal = observed_signal_ranks(views, observed)
    reduced = reduce_views(views, observed, signal)
    if rank_rule == "native":
        from native_rank import joint_rank_random_direction

        k = joint_rank_random_direction(views, observed, signal, seed=seed)
    elif rank_rule == "ajive":
        k = joint_rank_by_ajive(views, observed, signal)
    else:
        k = select_joint_rank(reduced, seed=seed)
    if k == 0:
        return None
    # Every view must be able to carry all k joint directions.
    signal = [max(s, k) for s in signal]
    reduced = reduce_views(views, observed, signal)
    model = IncompleteGCCA(n_components=k, shrinkage=0.0).fit(reduced)
    joint = model.transform_shared(reduced)
    individual = []
    for x, o, s in zip(views, observed.T, signal):
        xo = x[o] - x[o].mean(0)
        q = np.linalg.qr(joint[o] - joint[o].mean(0))[0]
        u, sv, _ = np.linalg.svd(xo - q @ (q.T @ xo), full_matrices=False)
        r = max(s - k, 0)
        scores = np.zeros((len(x), r))
        scores[o] = u[:, :r] * sv[:r]
        individual.append(scores)
    return joint, individual, dict(signal=signal, joint=k)
