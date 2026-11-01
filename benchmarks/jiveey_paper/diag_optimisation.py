"""Is JIVEEY's shortfall in the hardest cell optimisation or statistics?

Compare the loss and the shared recovery of the default fit (3 random starts), a 20-start fit,
and a fit started from AJIVE's solution (view weights regressed on AJIVE's scores).

python diag_optimisation.py
"""

from __future__ import annotations

import numpy as np
from common import JIVEEY, N_SHARED, REGIMES, ajive_fit, mean_filled, subspace_recovery
from scipy.optimize import minimize
from simulate import simulate

for regime, rate in (("fainter", 0.3), ("faint", 0.5)):
    design = REGIMES[regime]
    out = {k: [] for k in ("3 starts", "20 starts", "from AJIVE")}
    for seed in range(8):
        ds = simulate(seed, n_shared=N_SHARED, missing_rate=rate, **design)
        v, r = ds.masked, design["n_private"]
        for name, n_init in (("3 starts", 3), ("20 starts", 20)):
            m = JIVEEY(N_SHARED, r, 0.5).fit(v, seed=seed, n_init=n_init)
            out[name].append(
                (m.loss_, subspace_recovery(m.transform_joint(v), ds.shared_factors))
            )
        a = ajive_fit(v, ds.observed, N_SHARED, r)
        m = JIVEEY(N_SHARED, r, 0.5)
        m._prepare(v)
        filled = mean_filled(v, ds.observed)
        per_view = []
        for i, x in enumerate(m.x_):
            w = np.zeros((x.shape[1], N_SHARED + sum(m.r_)))
            joint = a.transform(filled)[i]
            w[:, :N_SHARED] = np.linalg.lstsq(x, joint, rcond=None)[0] if False else x.T @ joint / len(x)
            ind = a.individual_scores_[i]
            w[:, m.offset_[i] : m.offset_[i] + r] = x.T @ ind / len(x)
            per_view.append(w)
        res = minimize(m.loss_and_gradient, m._flatten(per_view), jac=True, method="L-BFGS-B", options={"maxiter": 1000})
        m.weights_ = m._factor(res.x)
        out["from AJIVE"].append(
            (res.fun, subspace_recovery(m.transform_joint(v), ds.shared_factors))
        )
    for name, vals in out.items():
        a_ = np.array(vals)
        print(f"{regime} {rate:.0%} {name:<11} loss {a_[:, 0].mean():9.3f}  shared {a_[:, 1].mean():.2f}")
