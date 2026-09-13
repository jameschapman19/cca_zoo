# GAM Methods

The `cca_zoo.gam` module provides `GAMCCA`, a nonlinear multiview CCA method that uses a
generalized additive model (GAM) — one smooth univariate B-spline per input feature — as the
per-view encoder. It has no optional dependency: the spline basis is built entirely from
scikit-learn's own `SplineTransformer`, with `scipy.optimize` doing the Newton solve, all already
required by `cca_zoo`, rather than reimplemented from scratch.

---

## Background

`GAMCCA` minimises the same unconstrained Eckart-Young (EY) objective used by the stochastic
`*_EY` models in `cca_zoo.linear`, by `DCCAEY` in `cca_zoo.deep`, and by `TreeCCA` in
`cca_zoo.tree` (the numpy-based models share the exact same implementation, in
`cca_zoo._utils._ey`):

$$
\mathcal{L}_{EY} = -2 \operatorname{tr}(C) + \operatorname{tr}(V V)
$$

where, for embeddings $Z_i = f_i(X_i)$, $C$ is the mean pairwise cross-covariance (including
$i = j$ terms) and $V$ the mean auto-covariance across all views. `GAMCCA` uses a generalized
additive model — $f_i(x) = \sum_j s_j(x_j)$, one B-spline term per input feature — in place of a
linear map (`CCAEY`) or a boosted-tree ensemble (`TreeCCA`) as the function class for each
$f_i$.

Writing $f_i(x) = \sum_j s_j(x_j)$ as a fixed per-feature B-spline basis times a coefficient
matrix, fitting those coefficients is a **P-IRLS** recipe — the same iteration structure GAM
software such as R's `mgcv` uses, applied directly to the EY loss rather than a per-observation
likelihood: for one latent component's coefficients at a time (every other component and view
held fixed), a damped Newton step is solved via `scipy.optimize.minimize(method="trust-exact")`
using the EY loss's exact gradient and Hessian in that coefficient space, cycling through every
component and view until the penalised objective stops moving.

Because each latent component decomposes exactly into one additive term per input feature, the
fitted shape of any feature's contribution is available directly via `model.shape_function(...)`
— the GAM analogue of `TreeCCA`'s split-gain feature importance, but an exact curve rather than a
single importance score, and smooth by construction rather than a step function.

**When to use:** Nonlinear multiview CCA where the true per-feature relationship is expected to
be *smooth* (rather than needing feature interactions or sharp thresholds) — a GAM's smoothness
assumption is then a genuine inductive-bias advantage, not just a cosmetic one.

If cross-view structure instead depends on an *interaction* between two features of the same view
(e.g. $x_1 x_2$), a GAM's additive structure cannot represent that the way a tree's multivariate
splits or a Gaussian process's joint kernel can — prefer `TreeCCA` or `GaussianProcessCCA` there.

---

## Basic usage

```python
from cca_zoo.gam import GAMCCA

model = GAMCCA(latent_dimensions=2).fit([X1, X2])
z1, z2 = model.transform([X1, X2])
corrs = model.score([X1, X2])

# GAMCCA also supports more than two views
model3 = GAMCCA(latent_dimensions=2).fit([X1, X2, X3])
```

## Inspecting fitted shape functions

`GAMCCA` has no linear weight matrices, so `model.weights` raises `NotImplementedError`. Use
`shape_function` instead to evaluate a single feature's fitted additive term directly:

```python
import numpy as np

model = GAMCCA(latent_dimensions=1).fit([X1, X2])

x_grid = np.linspace(X1[:, 0].min(), X1[:, 0].max(), 200)
shape = model.shape_function(view=0, feature=0, x=x_grid)  # (200, 1)
```

`shape` is that feature's contribution alone, in the units of the latent component — plot it
against `x_grid` to see the exact fitted curve for that feature, rather than a single importance
score. Summing every feature's `shape_function` at the training values reproduces
`model.encoders_[view].predict()` exactly.

---

## Key parameters

| Parameter | Description |
|---|---|
| `n_knots` | Knots per feature's B-spline term, passed straight through to `SplineTransformer(n_knots=...)`. More knots allow wigglier per-feature curves. |
| `alpha` | Ridge (smoothing) penalty strength applied to every spline coefficient. There is no automatic smoothing-parameter selection — tune this directly. |
| `max_iter` | Maximum number of full P-IRLS sweeps (one Newton solve per view and component each). |
| `tol` | Convergence tolerance on the penalised objective's change between consecutive sweeps. |
| `random_state` | Seed for the initial coefficients. |

---

## Practical notes

- `GAMCCA` supports 2 or more views.
- `latent_dimensions` must not exceed the number of features in any view.
- Unlike `KCCA`, `GAMCCA` does not store the training data for inference — new data is passed
  directly through the fitted per-feature splines, so `transform` on held-out data is inexpensive.
- No optional dependency is required (unlike `cca_zoo.tree`, which needs `xgboost`/`lightgbm`):
  `GAMCCA` is built entirely on `scikit-learn`'s `SplineTransformer` and `scipy.optimize`.
