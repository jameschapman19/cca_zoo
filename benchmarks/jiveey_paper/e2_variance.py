"""E2: variance explained, in sample and cross-fitted, against the simulator's truth.

python e2_variance.py [--seeds 20] [--workers 4]
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool

import numpy as np
from common import HERE, N_SHARED, REGIMES, JIVEEY, _folds, ajive_fit
from simulate import simulate

DESIGNS = ("moderate", "faint")


def cross_fitted_shares(views, k, r, c, seed) -> np.ndarray:
    """Held-out joint, individual and noise shares per view (views by 3), averaged over folds.

    Each fold's weights are fitted on the rest. Per view, the data are regressed on the
    training joint scores, then on the training individual scores with the joint span
    removed; the coefficients are applied to the held-out scores and the held-out variance
    they leave unexplained is the noise share.
    """
    out = []
    for train, test in _folds(views, seed):
        model = JIVEEY(k, r, c).fit([v[train] for v in views], seed=seed)
        tr_views, te_views = [v[train] for v in views], [v[test] for v in views]
        z_tr, z_te = model._projections(tr_views), model._projections(te_views)
        shares = []
        for i, (zt, zs, vt, vs, mean) in enumerate(
            zip(z_tr, z_te, tr_views, te_views, model.means_)
        ):
            o_tr, o_te = ~np.isnan(vt).any(1), ~np.isnan(vs).any(1)
            xt, xs = (vt - mean)[o_tr], (vs - mean)[o_te]
            jt, js = zt[o_tr][:, :k], zs[o_te][:, :k]
            off = model.offset_[i]
            it, is_ = zt[o_tr][:, off : off + r], zs[o_te][:, off : off + r]
            coef = np.linalg.lstsq(jt, it, rcond=None)[0]
            it, is_ = it - jt @ coef, is_ - js @ coef
            total = np.sum(xs**2)
            b_j = np.linalg.lstsq(jt, xt, rcond=None)[0]
            joint = 1 - np.sum((xs - js @ b_j) ** 2) / total
            both_t, both_s = np.hstack([jt, it]), np.hstack([js, is_])
            b = np.linalg.lstsq(both_t, xt, rcond=None)[0]
            explained = 1 - np.sum((xs - both_s @ b) ** 2) / total
            shares.append([joint, explained - joint, 1 - explained])
        out.append(shares)
    return np.mean(out, axis=0)


def run(job: tuple[str, float, int]) -> list[dict]:
    """Truth, in-sample JIVEEY, cross-fitted JIVEEY and AJIVE on one dataset."""
    regime, rate, seed = job
    design = REGIMES[regime]
    ds = simulate(seed, n_shared=N_SHARED, missing_rate=rate, **design)
    views, r = ds.masked, design["n_private"]
    base = dict(regime=regime, rate=rate, seed=seed)
    rows = [
        dict(
            **base,
            method="truth",
            joint=design["shared"],
            individual=design["private"],
            noise=1 - design["shared"] - design["private"],
        )
    ]
    model = JIVEEY(N_SHARED, r, 0.5).fit(views, seed=seed)
    v = model.variance_explained(views)
    rows.append(
        dict(
            **base,
            method="JIVEEY, in sample",
            joint=float(v["joint_total"].mean()),
            individual=float(v["individual_total"].mean()),
            noise=float(v["noise"].mean()),
        )
    )
    j, i, n = cross_fitted_shares(views, N_SHARED, r, 0.5, seed).mean(axis=0)
    rows.append(
        dict(**base, method="JIVEEY, cross-fitted", joint=j, individual=i, noise=n)
    )
    a = ajive_fit(views, ds.observed, N_SHARED, r)
    joint = float(np.mean([np.sum(x) for x in a.joint_variance_explained_]))
    indiv = float(np.mean([np.sum(x) for x in a.individual_variance_explained_]))
    rows.append(
        dict(**base, method="AJIVE, in sample", joint=joint, individual=indiv, noise=1 - joint - indiv)
    )
    return rows


def main() -> None:
    """Run and save."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    jobs = [(g, rate, s) for g in DESIGNS for rate in (0.0, 0.3) for s in range(args.seeds)]
    with Pool(args.workers) as pool:
        rows = [row for out in pool.imap_unordered(run, jobs) for row in out]
    (HERE / "output" / "e2.json").write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
