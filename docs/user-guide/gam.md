# GAM Methods

The `cca_zoo.gam` module provides `GAMCCA`, a nonlinear multiview CCA method that uses a
generalized additive model (GAM) — one smooth univariate B-spline per input feature — as the
per-view encoder. It has no optional dependency: the spline basis and its penalised fit are built
entirely from scikit-learn's own `SplineTransformer`, `Ridge` and `RidgeCV`, all already required
by `cca_zoo`, rather than reimplemented from scratch.

---

## Background

`GAMCCA` maximises the same unconstrained Eckart-Young (EY) objective used by the stochastic
`*_EY` models in `cca_zoo.linear`, by `DCCA_EY` in `cca_zoo.deep`, and by `TreeCCA` in
`cca_zoo.tree` (the numpy-based models share the exact same implementation, in
`cca_zoo._utils._ey`):

$$
\mathcal{L}_{EY} = -2 \operatorname{tr}(C) + \operatorname{tr}(V V)
$$

where, for embeddings $Z_i = f_i(X_i)$, $C$ is the mean pairwise cross-covariance (including
$i = j$ terms) and $V$ the mean auto-covariance across all views. `GAMCCA` uses a generalized
additive model — $f_i(x) = \sum_j s_j(x_j)$, one B-spline term per input feature — in place of a
linear map (`CCA_EY`) or a boosted-tree ensemble (`TreeCCA`) as the function class for each
$f_i$.

Unlike those models, which reach the EY loss's optimum by many small (stochastic-)gradient steps,
`GAMCCA` fits it the way GAM software such as R's `mgcv` fits an ordinary GAM, applied directly to
the EY loss instead of a per-observation likelihood:

1. **Inner loop — P-IRLS.** For the *current, fixed* smoothing parameters, repeatedly take a
   Newton step on the EY loss for each view in turn: form a working response
   $Z_i - \nabla_i / h_i$ from the analytic EY gradient $\nabla_i$ and a diagonal Hessian weight
   $h_i$, and ridge-regress it onto that view's fixed B-spline basis (`sklearn.linear_model.Ridge`).
   Cycle through every view until the EY loss itself stops moving.
2. **Outer loop — GCV-style smoothing-parameter search.** Only once the inner loop has converged,
   re-select each view's smoothing parameter with `sklearn.linear_model.RidgeCV`'s efficient
   leave-one-out cross-validation (the same statistical job GCV/REML do in `mgcv`) at that
   converged state, then re-run the inner loop with the new parameters. Repeat until the smoothing
   parameters stabilise too.

This is a deliberate departure from `TreeCCA`'s boosting recipe, which exists because a tree
ensemble has no closed-form fit to a moving target and so *must* be built up from many small
shrunk steps. A ridge-penalised spline fit has no such constraint, so `GAMCCA` has no
`learning_rate` or `n_estimators` to tune — each view's smoothing strength is chosen
automatically, exactly as it would be fitting any other GAM.

Because each latent component decomposes exactly into one additive term per input feature, the
fitted shape of any feature's contribution is available directly via `model.shape_function(...)`
— the GAM analogue of `TreeCCA`'s split-gain feature importance, but an exact curve rather than a
single importance score, and smooth by construction rather than a step function.

**When to use:** Nonlinear multiview CCA where the true per-feature relationship is expected to
be *smooth* (rather than needing feature interactions or sharp thresholds) — a GAM's smoothness
assumption is then a genuine inductive-bias advantage, not just a cosmetic one. On such data
`GAMCCA` reaches a higher held-out canonical correlation than `TreeCCA`, converges without any
round-count tuning, and is markedly cheaper per fit (a ridge solve is far cheaper than growing a
tree ensemble). See
`tests/gam/test_gamcca.py::test_gamcca_outperforms_linear_and_tree_on_smooth_nonmonotonic_data`
for a worked example: view 1 is a noisy linear copy of a shared latent factor `z`; view 2 is a
noisy linear copy of `z ** 2` (a smooth but non-monotonic transform, so no linear combination of
either view's raw features can align with the other — `rCCA` gets essentially nothing). `GAMCCA`
reaches a held-out canonical correlation of about 0.94, versus about 0.60 for `TreeCCA` at 150
boosting rounds (`TreeCCA` needs 600+ rounds to approach 0.90).

If cross-view structure instead depends on an *interaction* between two features of the same view
(e.g. $x_1 x_2$), a GAM's additive structure cannot represent that the way a tree's multivariate
splits can — prefer `TreeCCA` there.

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
| `alphas` | Candidate smoothing parameters searched by `RidgeCV` in the outer loop. Default is `numpy.logspace(-6, 3, 10)`. |
| `max_inner_iter` | Cap on P-IRLS rounds (one cycle through every view) per outer iteration. In practice the inner loop converges well before this in typical cases. |
| `max_outer_iter` | Cap on smoothing-parameter re-selection rounds. |
| `tol` | Inner-loop convergence tolerance, on the change in the EY loss between successive full passes over all views. |
| `hess_floor_percentile` | Percentile (0-100) of each round's raw diagonal-Hessian values used to floor them — self-calibrating damping against the Hessian's poor conditioning near the loss's own fixed point (see the class docstring for why). Default is 90. |
| `random_state` | Seed for the random-orthogonal initial embedding. |

Because the smoothing parameter is selected automatically, `GAMCCA` needs far less manual
hyperparameter search than `TreeCCA`; `n_knots` and `hess_floor_percentile` are the two knobs
worth adjusting if the defaults underperform, and both are stable across a wide range of data
sizes and scales in practice.

---

## Practical notes

- `GAMCCA` supports 2 or more views.
- `latent_dimensions` must not exceed the number of features in any view (the random-orthogonal
  initialisation draws that many orthogonal directions in feature space).
- Unlike `KCCA`, `GAMCCA` does not store the training data for inference — new data is passed
  directly through the fitted per-feature splines, so `transform` on held-out data is inexpensive.
- No optional dependency is required (unlike `cca_zoo.tree`, which needs `xgboost`/`lightgbm`):
  `GAMCCA` is built entirely on `scikit-learn`'s `SplineTransformer`, `Ridge` and `RidgeCV`.
