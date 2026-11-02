"""Mean held-out loss over (K, r) at 30% missing, 2% shared (true K=2, r=3), 8 datasets.

python diag_landscape.py
"""

from itertools import product
from multiprocessing import Pool

import numpy as np
from common import JIVEEY, N_SHARED, REGIMES, _folds, heldout_loss
from simulate import simulate

KS, RS = (0, 1, 2, 3, 4), (0, 1, 2, 3, 4, 5)


def run(seed):
    ds = simulate(seed, n_shared=N_SHARED, missing_rate=0.3, **REGIMES["faint"])
    v = ds.masked
    out = np.zeros((len(KS), len(RS)))
    for (a, k), (b, r) in product(enumerate(KS), enumerate(RS)):
        if k == 0 and r == 0:
            continue
        losses = []
        for train, test in _folds(v, seed):
            m = JIVEEY(k, r, 0.5).fit([x[train] for x in v], seed=seed, n_init=1)
            losses.append(heldout_loss(m, [x[test] for x in v]))
        out[a, b] = np.mean(losses)
    return out


if __name__ == "__main__":
    with Pool(4) as p:
        mean = np.mean(p.map(run, range(8)), axis=0)
    print("rows K=0..4, cols r=0..5"); print(np.round(mean, 2))
