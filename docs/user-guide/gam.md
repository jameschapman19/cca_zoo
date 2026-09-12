# GAM Methods

The `cca_zoo.gam` module provides `GAMCCA`, a nonlinear multiview CCA method that uses a
generalized additive model (GAM) — one smooth univariate B-spline per input feature — as the
per-view encoder. It has no optional dependency: the spline basis and its penalised fit are built
entirely from scikit-learn's own `SplineTransformer` and `Ridge`, both already required by
`cca_zoo`, rather than reimplemented from scratch.

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
$f_i$. The per-feature basis comes from `sklearn.preprocessing.SplineTransformer` (one contiguous
block of B-spline columns per feature, giving the additive structure directly) and each round's
penalised fit from `sklearn.linear_model.Ridge` — both scikit-learn's own, already-required
implementations, not reimplemented here.

Training proceeds by alternating (Gauss-Seidel) L2Boosting (Bühlmann & Yu, 2003): each round, for
every view in turn, the EY-loss gradient is computed from the current embeddings, rescaled to a
fixed target standard deviation (the analytic gradient's natural scale is far smaller than a
well-conditioned regression target — the same fix `TreeCCA` applies to its own boosters), and
ridge-fit onto that view's fixed spline basis. The fit is shrunk by `learning_rate` and added to a
running total. With `gauss_seidel=True` (the default) the gradient is recomputed from the
freshest embeddings before moving to the next view. Encoders start from a random-orthogonal,
unit-variance initial embedding per view, exactly as `TreeCCA` does.

Because each latent component decomposes exactly into one additive term per input feature, the
fitted shape of any feature's contribution is available directly via `model.shape_function(...)`
— the GAM analogue of `TreeCCA`'s split-gain feature importance, but an exact curve rather than a
single importance score, and smooth by construction rather than a step function.

**When to use:** Nonlinear multiview CCA where the true per-feature relationship is expected to
be *smooth* (rather than needing feature interactions or sharp thresholds) — a GAM's smoothness
assumption is then a genuine inductive-bias advantage, not just a cosmetic one. On such data
`GAMCCA` can reach a given held-out canonical correlation in far fewer boosting rounds than
`TreeCCA`, and generalise better at a matched round budget, because its basis represents a smooth
curve (e.g. a parabola) directly instead of approximating it with many small steps. See
`tests/gam/test_gamcca.py::test_gamcca_outperforms_linear_and_tree_on_smooth_nonmonotonic_data`
for a worked example: view 1 is a noisy linear copy of a shared latent factor `z`; view 2 is a
noisy linear copy of `z ** 2` (a smooth but non-monotonic transform, so no linear combination of
either view's raw features can align with the other — `rCCA` gets essentially nothing). At a
matched budget of 150 boosting rounds, `GAMCCA` reaches a held-out canonical correlation of about
0.97, versus about 0.60 for `TreeCCA` — which does not reach that level even at 600 rounds (about
0.90) — and `GAMCCA` is also markedly cheaper per round, since a ridge-regularised least-squares
solve is far cheaper than growing a tree.

If cross-view structure instead depends on an *interaction* between two features of the same view
(e.g. $x_1 x_2$), a GAM's additive structure cannot represent that the way a tree's multivariate
splits can — prefer `TreeCCA` there.

---

## Basic usage

```python
from cca_zoo.gam import GAMCCA

model = GAMCCA(latent_dimensions=2, n_estimators=150).fit([X1, X2])
z1, z2 = model.transform([X1, X2])
corrs = model.score([X1, X2])

# GAMCCA also supports more than two views
model3 = GAMCCA(latent_dimensions=2, n_estimators=150).fit([X1, X2, X3])
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
score.

---

## Key parameters

| Parameter | Description |
|---|---|
| `n_estimators` | Boosting rounds. Higher values fit more complex relationships but risk overfitting and cost more time. |
| `n_knots` | Knots per feature's B-spline term, passed straight through to `SplineTransformer(n_knots=...)`. More knots allow wigglier per-feature curves. |
| `learning_rate` | Boosting shrinkage applied to each round's ridge fit. |
| `ridge` | Ridge penalty for each round's per-view spline fit, passed straight through to `Ridge(alpha=...)`. The main defence against overfitting a single view's noise — increase it for smoother, less wiggly curves. |
| `gauss_seidel` | Use freshly-updated view-1 embeddings when computing view 2's gradient each round (default `True`); set `False` for Jacobi-style stale updates. |
| `random_state` | Seed for the random-orthogonal initial embedding. |

Hyperparameters are best selected by cross-validation with `GridSearchCV` from
`cca_zoo.model_selection`, as for other models.

---

## Practical notes

- `GAMCCA` supports 2 or more views.
- `latent_dimensions` must not exceed the number of features in any view (the random-orthogonal
  initialisation draws that many orthogonal directions in feature space).
- Unlike `KCCA`, `GAMCCA` does not store the training data for inference — new data is passed
  directly through the fitted per-feature splines, so `transform` on held-out data is inexpensive.
- No optional dependency is required (unlike `cca_zoo.tree`, which needs `xgboost`/`lightgbm`):
  `GAMCCA` is built entirely on `scikit-learn`'s `SplineTransformer` and `Ridge`.
