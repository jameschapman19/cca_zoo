# Isotonic Methods

The `cca_zoo.isotonic` module provides `IsotonicCCA`, a nonlinear multiview CCA method that uses
a monotonic additive model — one univariate isotonic (monotonic step) term per input feature,
fit via PAVA — as the per-view encoder. It has no optional dependency: built entirely on
scikit-learn's own `IsotonicRegression`, already required by `cca_zoo`.

---

## Background

`IsotonicCCA` minimises the same unconstrained Eckart-Young (EY) objective used throughout
`cca_zoo` (see `cca_zoo._utils._ey`):

$$
\mathcal{L}_{EY} = -2 \operatorname{tr}(C) + \operatorname{tr}(V V)
$$

where, for embeddings $Z_i = f_i(X_i)$, $C$ is the mean pairwise cross-covariance (including
$i = j$ terms) and $V$ the mean auto-covariance across all views. Like `GAMCCA`, $f_i$ is
additive across features, $f_i(x) = \sum_j s_j(x_j)$ — the difference is the shape each term
$s_j$ is allowed to take: here it's constrained to be **monotonic**, fit via
`sklearn.isotonic.IsotonicRegression` (PAVA) rather than an unconstrained B-spline.

An isotonic step function has no fixed finite basis (its breakpoints adapt to the data, the same
way a decision tree's splits do), so unlike `GAMCCA`'s single joint Newton-CG solve, `IsotonicCCA`
is fit by the same **functional gradient boosting** recipe `TreeCCA` uses: each round, for every
view in turn, one isotonic term is fit per feature against that view's current EY gradient and
added (shrunk by `learning_rate`) to that feature's running contribution — Gauss-Seidel across
views, exactly as in `TreeCCA`. It's `TreeCCA`'s boosting recipe crossed with `GAMCCA`'s additive,
per-feature, inspectable structure.

Each feature's monotonicity direction is fixed once (from its correlation with the first round's
descent direction) and reused for every later round: a sum of same-direction monotonic terms is
itself guaranteed monotonic, which is what makes the *whole* fitted curve monotonic, not just one
round of it.

**When to use:** Nonlinear multiview CCA where a feature's effect is expected to be monotonic —
interpretability under that constraint is the entire point. `GAMCCA`'s `shape_function` shows an
exact curve with no shape constraint at all; `IsotonicCCA`'s shows one that's provably monotonic
throughout, at the cost of being unable to represent a genuinely non-monotonic effect (a feature
with a U-shaped or periodic influence needs `GAMCCA`, `TreeCCA`, or `GaussianProcessCCA` instead).

As with `GAMCCA`, an additive encoder can't represent a genuine *interaction* between two features
of the same view — reach for `TreeCCA` or `GaussianProcessCCA` if cross-view structure only shows
up through such interactions.

---

## Basic usage

```python
from cca_zoo.isotonic import IsotonicCCA

model = IsotonicCCA(latent_dimensions=2).fit([X1, X2])
z1, z2 = model.transform([X1, X2])
corrs = model.score([X1, X2])

# IsotonicCCA also supports more than two views
model3 = IsotonicCCA(latent_dimensions=2).fit([X1, X2, X3])
```

## Inspecting fitted shape functions

`IsotonicCCA` has no linear weight matrices, so `model.weights` raises `NotImplementedError`. Use
`shape_function` instead to evaluate a single feature's fitted additive term directly:

```python
import numpy as np

model = IsotonicCCA(latent_dimensions=1).fit([X1, X2])

x_grid = np.linspace(X1[:, 0].min(), X1[:, 0].max(), 200)
shape = model.shape_function(view=0, feature=0, x=x_grid)  # (200, 1), guaranteed monotonic
```

`shape` is that feature's contribution alone, in the units of the latent component. Summing every
feature's `shape_function` at the training values reproduces `model.encoders_[view].predict()`
exactly.

---

## Key parameters

| Parameter | Description |
|---|---|
| `n_estimators` | Number of boosting rounds. |
| `learning_rate` | Shrinkage applied to each round's fitted isotonic terms. Lower than `TreeCCA`'s default: each round here fits and sums *one term per feature*, not one weak learner total, so the effective per-round step already scales with the view's width. |
| `subsample` | Row fraction (without replacement) used to fit each round's isotonic terms. Not optional the way it is for `TreeCCA`/`GAMCCA`: an isotonic fit has no built-in capacity limit, so this is what keeps many-round boosting from overfitting. |
| `out_of_bounds` | How each fitted term extrapolates beyond its training range (`sklearn.isotonic.IsotonicRegression`'s own parameter). `"clip"` (default) holds the boundary value constant. |
| `gauss_seidel` | Whether to re-predict a view's embedding (and recompute the gradient) immediately after updating it, before moving to the next view, vs. using the same stale embeddings for every view's gradient within a round. |
| `random_state` | Seed for the random-orthogonal initial embedding (isotonic regression itself is deterministic). |

---

## Practical notes

- `IsotonicCCA` supports 2 or more views.
- `latent_dimensions` must not exceed the number of features in any view.
- No optional dependency is required (unlike `cca_zoo.tree`, which needs `xgboost`/`lightgbm`):
  `IsotonicCCA` is built entirely on scikit-learn's `IsotonicRegression`.
