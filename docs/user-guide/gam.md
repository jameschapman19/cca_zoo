# GAM & MARS Methods

The `cca_zoo.gam` module provides two spline-based nonlinear multiview CCA methods: `GAMCCA` and
`MARSCCA` (see [below](#marscca)). `GAMCCA` is a nonlinear multiview CCA method that uses a
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
matrix, fitting those coefficients is conceptually a **P-IRLS** recipe — the same iteration
structure GAM software such as R's `mgcv` uses, applied directly to the EY loss rather than a
per-observation likelihood. Rather than hand-rolling that solve (or even cycling over views one
at a time), every view's coefficients — every latent component, every view, all at once — are
updated in a single call to `scipy.optimize.minimize(method="trust-krylov")`, a standard
off-the-shelf trust-region Newton-CG solver, given the EY loss's exact gradient and an exact
Hessian-vector product across the whole stacked parameter vector.

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
| `max_iter` | Maximum number of outer Newton iterations in the single joint `trust-krylov` solve (`scipy.optimize.minimize`'s own `maxiter` option). |
| `tol` | Gradient-norm convergence tolerance for the joint solve (`scipy.optimize.minimize`'s own `gtol` option). |
| `random_state` | Seed for the initial coefficients. |

---

## Practical notes

- `GAMCCA` supports 2 or more views.
- `latent_dimensions` must not exceed the number of features in any view.
- Unlike `KCCA`, `GAMCCA` does not store the training data for inference — new data is passed
  directly through the fitted per-feature splines, so `transform` on held-out data is inexpensive.
- No optional dependency is required (unlike `cca_zoo.tree`, which needs `xgboost`/`lightgbm`):
  `GAMCCA` is built entirely on `scikit-learn`'s `SplineTransformer` and `scipy.optimize`.

---

## MARSCCA

`MARSCCA` swaps `GAMCCA`'s fixed B-spline basis for a multivariate adaptive regression spline
(Friedman, 1991): each view's encoder is a linear combination of basis functions, each a product
of up to `max_degree` hinges $\max(0, \pm(x_j - t))$, and the basis is *grown* rather than fixed.

Every forward step scores each candidate reflected hinge pair — any existing term (or the
constant) as parent, any feature not already in that parent, any of `n_candidate_knots` interior
quantile knots — by how much of the current EY gradient the pair can absorb once orthogonalised
against the current basis. That is classical MARS's residual-sum-of-squares criterion with the
residual replaced by the EY loss's negative gradient. The best pair is added to each view in
turn, then every view's coefficients are refit jointly. On a fixed basis the ridge-EY fit is a
generalized eigenproblem — the one ridge-regularised MCCA solves — so each refit is its exact
global optimum in closed form. Knots therefore land only where the cross-view signal needs them.

With `max_degree=1` (the default, as in R's `earth`) the encoder is additive, like `GAMCCA` but
with adaptive knots. With `max_degree=2` a term can represent a within-view interaction such as
$x_1 x_2$ — exactly the case the note above says `GAMCCA` cannot handle — while staying
inspectable term by term.

```python
from cca_zoo.gam import MARSCCA

model = MARSCCA(latent_dimensions=1, max_degree=2, max_terms=20).fit([X1, X2])
terms = model.basis_functions(0)  # e.g. ['h(x1 - 0.41)', 'h(0.41 - x1)', ...]
coefs = model.encoders_[0].coef_  # (n_terms, latent_dimensions), row m ↔ terms[m]
```

Products appear as e.g. `'h(x1 - 0.41) * h(x0 + 0.2)'`, with `h(u) = max(0, u)` and knots in the
raw feature units.

### Pruning

The forward pass deliberately overshoots, so, as in R's `earth`, a backward pass prunes it: from
every term the forward pass added, it repeatedly deletes the term (from whichever view) whose
removal raises the refit training EY loss least, down to `n_terms` terms in total. Every refit
being a closed-form eigenproblem, each deletion is exact, and every candidate is scored at once
by one batched eigenvalue decomposition. Unlike truncating the forward sequence, the backward
pass can drop a stepping-stone term — a lone hinge in $x_1$, say — once the interaction it led
to has taken over its job.

`earth` then picks the size by GCV, a squared-error quantity with no EY-loss counterpart. Its
alternative, choosing the size along the backward sequence by cross-validation
(`pmethod="cv"`), carries over exactly as a search over `n_terms`. Refit with
[`one_standard_error`](model-selection.md#preferring-simpler-models-one_standard_error) to take
the smallest model within one standard error of the best rather than the noisy maximum, which
on pure noise keeps dozens of terms:

```python
from cca_zoo.model_selection import GridSearchCV, one_standard_error

gs = GridSearchCV(
    MARSCCA(max_degree=2, max_terms=40),
    {"n_terms": [2, 4, 8, 12, 16, 24, 32, 48, 80]},
    refit=one_standard_error("n_terms"),
).fit([X1, X2])
```

| Parameter | Description |
|---|---|
| `max_terms` | Maximum basis functions per view in the forward pass (each step adds at most two). Scalar or per-view list. |
| `n_terms` | Total terms, across views, kept by the backward pass (each view keeps at least one); `None` keeps the whole forward pass. The parameter to search when pruning. |
| `max_degree` | Maximum hinge factors per basis function: 1 is additive, 2 allows pairwise interactions. Scalar or per-view list. |
| `n_candidate_knots` | Candidate knots per feature, at interior quantiles of the training values. |
| `alpha` | Ridge penalty on every basis coefficient. Scalar or per-view list. |
