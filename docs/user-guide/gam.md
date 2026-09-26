# GAM & MARS Methods

The `cca_zoo.gam` module provides two spline-based nonlinear multiview CCA methods, each modelled
on the R package its users will know: `GAMCCA` follows `mgcv`, and `MARSCCA` (see
[below](#marscca)) follows `earth`. Both are built on scikit-learn's `SplineTransformer` and
SciPy, with no optional dependency.

---

## GAMCCA

`GAMCCA` gives each view a generalized additive model — $f_i(x) = \sum_j s_{ij}(x_j)$, one smooth
per input feature — where each smooth is `mgcv`'s P-spline, `s(x, bs="ps", k=k, m=m)`: a
`k`-dimensional B-spline basis on evenly spaced knots with a difference penalty on neighbouring
coefficients (Eilers and Marx, 1996). The embeddings minimise the Eckart-Young (EY) objective
shared with `CCAEY`, `DCCAEY` and `TreeCCA`,

$$
\mathcal{L}_{EY} = -2 \operatorname{tr}(C) + \operatorname{tr}(V V)
$$

($C$ the mean pairwise cross-covariance including $i = j$ terms, $V$ the mean auto-covariance),
plus each view's smoothing penalty $\tfrac12\,\mathrm{sp}_i \sum_j \beta_{ij}^\top D^\top D
\beta_{ij}$. The penalty shrinks every smooth towards a polynomial — a straight line for the
default second-difference penalty — rather than towards zero.

With the basis fixed, the whole fit is one generalized eigenproblem, solved in closed form at its
global optimum: no iterations, no tolerance, no random start. It is the same solver `MARSCCA`
uses. A P-spline basis is rank-deficient by design — splines with no data under them are kept
for the penalty to fill in, and every feature's splines sum to a constant — so, as `mgcv`
resolves identifiability by reparametrisation, the problem is first rewritten exactly on the
basis's row space.

```python
from cca_zoo.gam import GAMCCA

model = GAMCCA(latent_dimensions=2).fit([X1, X2])
z1, z2 = model.transform([X1, X2])
corrs = model.score([X1, X2])

# mgcv-style controls, per view where useful
model = GAMCCA(k=[10, 30], m=(2, 1), sp=[0.1, 1.0]).fit([X1, X2])
```

### Choosing the smoothing parameter

`mgcv` estimates each `sp` by GCV or REML, both likelihood or residual criteria with no EY-loss
counterpart, so here `sp` is chosen by cross-validation. Refit with
[`one_standard_error`](model-selection.md#preferring-simpler-models-one_standard_error), telling it
that a larger `sp` is the simpler model, to take the smoothest fit within one standard error of
the best:

```python
from cca_zoo.model_selection import GridSearchCV, one_standard_error

gs = GridSearchCV(
    GAMCCA(),
    {"sp": [1e-3, 1e-2, 1e-1, 1, 10, 100]},
    refit=one_standard_error("sp", larger_is_simpler=True),
).fit([X1, X2])
```

### Inspecting fitted smooths

`GAMCCA` has no linear weight matrices, so `model.weights` raises `NotImplementedError`.
`shape_function` evaluates one feature's fitted smooth — `plot.gam`'s partial effect:

```python
import numpy as np

x_grid = np.linspace(X1[:, 0].min(), X1[:, 0].max(), 200)
shape = model.shape_function(view=0, feature=0, x=x_grid)  # (200, latent_dimensions)
```

Summing every feature's `shape_function` at the training values reproduces
`model.encoders_[view].predict()`.

### Parameters

Parameters share `mgcv`'s names and meanings for `bs="ps"` smooths.

| Parameter | `mgcv` | Description |
|---|---|---|
| `k` | `k` | Basis dimension: B-splines per feature. Default 10, `mgcv`'s; raise it when a relationship needs more wiggles, and the penalty keeps the larger basis in check. Cost grows with the cube of the total basis size. Scalar or per-view list. |
| `m` | `m` | `(order, penalty order)`: B-splines of degree `order + 1` and a `penalty order`-th difference penalty; a single value sets both. Default 2 (cubic, second differences), `mgcv`'s. An int or tuple, or a list of per-view values. |
| `sp` | `sp` | Smoothing parameter; larger is smoother. Default 0.01. `mgcv` estimates it; here choose it by cross-validation (above). Scalar or per-view list. |

**When to use:** nonlinear multiview CCA where each feature's relationship is expected to be
smooth. A GAM is additive, so it cannot represent an interaction between two features of the
same view (e.g. $x_1 x_2$); `MARSCCA` with `degree >= 2`, `TreeCCA` or `GaussianProcessCCA` can.

---

## MARSCCA

`MARSCCA` swaps `GAMCCA`'s fixed B-spline basis for a multivariate adaptive regression spline
(Friedman, 1991): each view's encoder is a linear combination of basis functions, each a product
of up to `degree` hinges $\max(0, \pm(x_j - t))$, and the basis is *grown* rather than fixed.

Every forward step scores each candidate reflected hinge pair — any existing term (or the
constant) as parent, any feature not already in that parent, any knot `earth`'s `minspan` and
`endspan` rules allow within the parent's support — by how much of the current EY gradient the
pair can absorb once orthogonalised against the current basis: classical MARS's
residual-sum-of-squares criterion with the residual replaced by the EY loss's negative gradient.
The best ten of those are then re-ranked by their exact loss after a refit, the criterion
`earth` applies to every candidate (scoring all of them exactly would take an eigenproblem
each). The best pair is added to each view in
turn, then every view's coefficients are refit jointly. On a fixed basis the ridge-EY fit is a
generalized eigenproblem — the one ridge-regularised MCCA solves — so each refit is its exact
global optimum in closed form. Knots therefore land only where the cross-view signal needs them.

With `degree=1` (the default, as in R's `earth`) the encoder is additive, like `GAMCCA` but
with adaptive knots. With `degree=2` a term can represent a within-view interaction such as
$x_1 x_2$ — exactly the case the note above says `GAMCCA` cannot handle — while staying
inspectable term by term.

```python
from cca_zoo.gam import MARSCCA

model = MARSCCA(latent_dimensions=1, degree=2, nk=20).fit([X1, X2])
terms = model.basis_functions(0)  # e.g. ['h(x1 - 0.41)', 'h(0.41 - x1)', ...]
coefs = model.encoders_[0].coef_  # (len(terms), latent_dimensions), row m ↔ terms[m]
```

Products appear as e.g. `'h(x1 - 0.41) * h(x0 + 0.2)'`, with `h(u) = max(0, u)` and knots in the
raw feature units.

### Pruning

The forward pass deliberately overshoots, so, as in R's `earth`, a backward pass prunes it: from
every term the forward pass added, it repeatedly deletes the term (from whichever view) whose
removal raises the refit training EY loss least, down to `nprune` terms in total. Every refit
being a closed-form eigenproblem, each deletion is exact: removing a term restricts that
eigenproblem by one linear constraint, so every candidate's new eigenvalues follow from one
eigendecomposition per step, the eigenvalue analogue of `earth`'s least-squares downdates.
Unlike truncating the forward sequence, the backward pass can drop a stepping-stone term — a
lone hinge in $x_1$, say — once the interaction it led to has taken over its job. The forward
pass itself also stops early, as `earth`'s does, once a round lowers the loss by less than a
fraction `thresh` (default 0.001).

`earth` then picks the size by GCV, a squared-error quantity with no EY-loss counterpart. Its
alternative, choosing the size along the backward sequence by cross-validation
(`pmethod="cv"`), carries over exactly as a search over `nprune`. Refit with
[`one_standard_error`](model-selection.md#preferring-simpler-models-one_standard_error) to take
the smallest model within one standard error of the best rather than the noisy maximum, which
on pure noise keeps dozens of terms:

```python
from cca_zoo.model_selection import GridSearchCV, one_standard_error

gs = GridSearchCV(
    MARSCCA(degree=2, nk=40),
    {"nprune": [2, 4, 8, 12, 16, 24, 32, 48, 80]},
    refit=one_standard_error("nprune"),
).fit([X1, X2])
```

### Variable importance

`model.variable_importance(criterion)` is `earth`'s `evimp`, computed over the backward pass's
nested subsets from the fitted model down: `"nsubsets"` counts the subsets that use each
feature, and `"loss"` credits each subset's decrease in EY loss over the next smaller one to
every feature it uses, scaled so the most important feature across views scores 100.

```python
importance = model.variable_importance("loss")  # one array per view
```

Parameters share `earth`'s names and defaults wherever `earth` has one. `earth` counts an
intercept in `nk` and `nprune`; views here are centred, so neither counts one. The one default
that differs from `earth`'s is `minspan` (see the table). As elsewhere in
the package, every parameter that configures one view's basis takes a single value or a list
with one entry per view (`None` entries take that view's default). `thresh` and `nprune` are
global: both are judged on the joint fit across views.

| Parameter | `earth` | Description |
|---|---|---|
| `degree` | `degree` | Maximum hinge factors per term: 1 is additive (default), 2 allows pairwise interactions. Scalar or per-view list. |
| `nk` | `nk` | Maximum terms per view in the forward pass. Default `min(200, max(20, 2 * n_features))`, `earth`'s less its intercept. Scalar or per-view list. |
| `nprune` | `nprune` | Total terms, across views, kept by the backward pass. `None` keeps the whole forward pass, since `earth`'s default of choosing it by GCV has no EY counterpart — search it by cross-validation instead. |
| `thresh` | `thresh` | Forward-pass stopping threshold (default 0.001): stop once a round lowers the loss by less than `thresh` times its magnitude. |
| `minspan`, `endspan` | `minspan`, `endspan` | Knot rules within each parent's support: at least `minspan` points between knots, none within `endspan` points of either end (doubled for interaction terms, as `Adjust.endspan=2`). `endspan=None` is Friedman's formula, `earth`'s default. `minspan=0` is Friedman's spacing exactly, as in `earth`; the default `None` widens it to leave at most 20 knots per feature, which markedly helps this forward pass find interactions (held-out 0.94 against 0.61 on a pure three-way interaction). Raising `minspan` is also the lever for speed on large data. Scalar or per-view list. |
| `alpha` | — | Ridge penalty on every basis coefficient (CCA needs it; the EY fit is otherwise unregularised). Scalar or per-view list. |
