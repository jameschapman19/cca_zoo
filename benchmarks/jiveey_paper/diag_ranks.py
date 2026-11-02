"""At 50% missing, does the held-out loss prefer the true joint rank on average (bias) or only noisily (variance)?

python diag_ranks.py
"""

import numpy as np
from common import JIVEEY, N_SHARED, REGIMES, _folds, heldout_loss
from simulate import simulate

d = REGIMES["faint"]
rows = {k: [] for k in (1, 2, 3)}
for seed in range(8):
    ds = simulate(seed, n_shared=N_SHARED, missing_rate=0.5, **d)
    v = ds.masked
    for k in rows:
        losses = []
        for train, test in _folds(v, seed, 5):
            m = JIVEEY(k, 3, 0.5).fit([x[train] for x in v], seed=seed, n_init=1)
            losses.append(heldout_loss(m, [x[test] for x in v]))
        rows[k].append(np.mean(losses))
arr = np.array([rows[k] for k in (1, 2, 3)])
print("mean held-out loss K=1,2,3:", arr.mean(1).round(3))
print("per-dataset argmin K:", (arr.argmin(0) + 1).tolist())
