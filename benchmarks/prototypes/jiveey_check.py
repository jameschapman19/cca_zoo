"""Check JIVEEY: its gradient, and that with no individual columns it is CCAEY.

python jiveey_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from jiveey import JIVEEY
from scipy.optimize import check_grad

sys.path.insert(0, str(Path(__file__).parent.parent / "missing_views"))
from simulate import simulate  # noqa: E402

from cca_zoo.linear import CCAEY  # noqa: E402


def principal_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    qa = np.linalg.qr(a - a.mean(axis=0))[0]
    qb = np.linalg.qr(b - b.mean(axis=0))[0]
    return np.linalg.svd(qa.T @ qb, compute_uv=False)


rng = np.random.default_rng(0)
views = [rng.standard_normal((50, p)) + 0.5 for p in (8, 6, 5)]
views[0][:10], views[2][40:] = np.nan, np.nan
for c in (0.0, 0.5):
    model = JIVEEY(2, [2, 1, 1], shrinkage=c)
    model._prepare(views)
    theta = (
        rng.standard_normal(sum(p * (2 + r) for p, r in zip(model.p_, model.r_))) * 0.3
    )
    f = lambda t: model.loss_and_gradient(t)[0]
    g = lambda t: model.loss_and_gradient(t)[1]
    print(
        f"gradient, c={c}, views missing: relative error {check_grad(f, g, theta) / np.linalg.norm(g(theta)):.1e}"
    )

# With no individual columns it is CCAEY: the same joint subspace on complete data.
ds = simulate(
    0,
    n_samples=150,
    widths=(60, 80, 40),
    n_shared=2,
    n_private=2,
    shared=0.2,
    private=0.4,
)
for c in (0.2, 0.5):
    ours = JIVEEY(2, 0, shrinkage=c).fit(ds.views).transform_joint(ds.views)
    ccaey = CCAEY(n_components=2, shrinkage=c, random_state=0).fit(ds.views)
    reference = np.mean(ccaey.transform(ds.views), axis=0)
    print(
        f"no individual columns, c={c}: principal cosines with CCAEY {np.round(principal_cosines(ours, reference), 4)}"
    )
