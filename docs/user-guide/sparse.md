# Sparse Methods

The `cca_zoo.sparse` module provides three sparse linear multiview CCA methods, each the EY-loss
analogue of a scikit-learn regularised linear regressor: `ElasticNetCCA` (elastic net),
`MultiTaskElasticNetCCA` (multi-task elastic net — row-group sparsity across latent dimensions),
and `OrthogonalMatchingPursuitCCA` (greedy fixed-cardinality selection). None has an optional
dependency: all are built entirely on numpy, already required by `cca_zoo`.

---

## Background

`ElasticNetCCA` minimises the same unconstrained Eckart-Young (EY) objective used by the
`CCAEY`/`PLSEY`/`StochasticCCAEY` models in `cca_zoo.linear`, by `DCCAEY` in `cca_zoo.deep`, by
`TreeCCA` in `cca_zoo.tree`, and by `GAMCCA`/`GaussianProcessCCA` (all share the exact same
implementation, in `cca_zoo._utils._ey`), with an elastic-net penalty added on the per-view
weights:

$$
\mathcal{L}(W) = \mathcal{L}_{EY}(Z_1, \dots, Z_M)
    + \sum_i \left( \alpha \, \rho \, \|W_i\|_1 + \tfrac{1}{2} \alpha (1-\rho) \|W_i\|_F^2 \right)
$$

where $Z_i = X_i W_i$ and $\rho$ is `l1_ratio`. `ElasticNetCCA` uses a plain linear map for each
$f_i$, like `CCAEY`, but reaches its optimum by **cyclic coordinate descent** instead of gradient
descent — the same algorithm `sklearn.linear_model.ElasticNet` uses for ordinary (squared-error)
elastic net, updating one scalar weight at a time to its exact minimiser with every other weight
held fixed.

This is a genuine departure from `ElasticNet`'s own coordinate descent, not a re-use of it: for
ordinary least squares, the loss restricted to a single coordinate is a plain quadratic, giving
the familiar closed-form soft-threshold update. Restricting $\mathcal{L}_{EY}$ to a single
coordinate instead gives an **exact quartic** (not quadratic), which each update solves to its
exact global minimiser directly — no line search, no step size, and no local-optimum risk from
linearising the loss.

Because every embedding stays exactly linear in the raw (centred) view throughout fitting,
`model.weights` returns real per-view canonical weight matrices, unlike `TreeCCA`, `GAMCCA`, and
`GaussianProcessCCA`, where it raises `NotImplementedError`.

**When to use:** Multiview CCA where the true relationship is linear but only a subset of
features in each view actually drive the shared structure — `l1_ratio > 0` drives irrelevant
features' weights to exactly zero, giving feature selection alongside the fitted canonical
directions. When every feature is expected to contribute, plain `CCAEY`/`rCCA` need less
hyperparameter tuning (no `alpha`/`l1_ratio` to select).

Passing `positive=True` additionally constrains every weight to be non-negative, mirroring
`sklearn.linear_model.Lasso`/`ElasticNet`'s own `positive=True`.

---

## MultiTaskElasticNetCCA: sparsity shared across latent dimensions

`ElasticNetCCA`'s penalty is per-scalar: a feature can easily survive in canonical component 1
while dropping out of component 2, for no principled reason. `MultiTaskElasticNetCCA` uses
`sklearn.linear_model.MultiTaskLasso`/`MultiTaskElasticNet`'s row-group penalty instead —
$\|W_i\|_{2,1} = \sum_j \|W_i[j,:]\|_2$, summing each *feature's* weight-row norm over every
latent dimension — so a feature is either active in every component or in none. That is arguably
a more natural fit for CCA than for ordinary multi-task regression, since a CCA model's latent
dimensions are not independent "tasks" fit separately but different views of the same features.

```python
from cca_zoo.sparse import MultiTaskElasticNetCCA

model = MultiTaskElasticNetCCA(latent_dimensions=2, alpha=0.1, l1_ratio=0.5).fit(
    [X1, X2]
)
active_features = (model.weights[0] != 0).any(axis=1)  # same mask for every component
```

Unlike `ElasticNetCCA`'s exact per-scalar quartic solve, a whole row's $k$ coefficients are
coupled by $\mathcal{L}_{EY}$'s auto-covariance cross terms, so there's no closed-form joint
minimiser for a row the way there is for a single scalar. Each row is instead updated by one step
of proximal gradient (ISTA) with backtracking line search, accepted only once verified to
decrease the exact penalised objective — so fitting is still provably monotonic, just without
`ElasticNetCCA`'s additional per-step exactness.

---

## OrthogonalMatchingPursuitCCA: a fixed sparsity budget

Where `ElasticNetCCA`/`MultiTaskElasticNetCCA` reach a sparsity level indirectly, by tuning a
continuous penalty strength, `OrthogonalMatchingPursuitCCA` (the EY-loss analogue of
`sklearn.linear_model.OrthogonalMatchingPursuit`) fixes it directly: specify how many features
each view is allowed to use, and features are added one at a time — by the same residual-
correlation criterion classical OMP uses — with the active coefficients re-solved to their exact
joint (unpenalised) optimum after every addition.

```python
from cca_zoo.sparse import OrthogonalMatchingPursuitCCA

model = OrthogonalMatchingPursuitCCA(latent_dimensions=2, n_nonzero_coefs=5).fit(
    [X1, X2]
)
active_features = (model.weights[0] != 0).any(axis=1)
assert active_features.sum() == 5
```

`n_nonzero_coefs` can be a single `int` (applied to every view) or a list (one budget per view);
left as `None`, each view defaults to `max(1, n_features_i // 10)`, mirroring sklearn's own
default.

---

## ElasticNetCCA usage

```python
from cca_zoo.sparse import ElasticNetCCA

model = ElasticNetCCA(latent_dimensions=2, alpha=0.1, l1_ratio=0.5).fit([X1, X2])
z1, z2 = model.transform([X1, X2])
corrs = model.score([X1, X2])

# ElasticNetCCA also supports more than two views
model3 = ElasticNetCCA(latent_dimensions=2, alpha=0.1).fit([X1, X2, X3])
```

`l1_ratio=0` is pure ridge (no sparsity); `l1_ratio=1` is pure lasso. Increasing `alpha` shrinks
more weights to exactly zero:

```python
model = ElasticNetCCA(latent_dimensions=1, alpha=0.5, l1_ratio=0.9).fit([X1, X2])
nonzero_features = (model.weights[0] != 0).any(axis=1)
```

---

## Key parameters

| Parameter | `ElasticNetCCA` | `MultiTaskElasticNetCCA` | `OrthogonalMatchingPursuitCCA` |
|---|---|---|---|
| Sparsity control | `alpha`, `l1_ratio` | `alpha`, `l1_ratio` | `n_nonzero_coefs` |
| `alpha` | Overall elastic-net penalty strength. | Overall row-group penalty strength. | — |
| `l1_ratio` | Mixing in `[0, 1]`; 0 ridge, 1 lasso. | Mixing in `[0, 1]`; 0 ridge, 1 row-group lasso. | — |
| `n_nonzero_coefs` | — | — | Target active-feature count, per view (`int`, list, or `None`). |
| `positive` | Constrain weights `>= 0`. | — | — |
| `max_iter` | Coordinate-descent sweeps. | Coordinate-descent sweeps. | Outer view-cycling rounds. |
| `tol` | Objective change tolerance between sweeps. | Same. | Same, and between per-addition refit sweeps. |
| `random_state` | Seed for the initial weights. | Same. | Seed for the initial (pre-selection) weights. |

Hyperparameters are best selected by cross-validation with `GridSearchCV` from
`cca_zoo.model_selection`, as for other models.

---

## Practical notes

- All three support 2 or more views.
- `latent_dimensions` must not exceed the number of features in any view.
- Like every EY-loss model, $\mathcal{L}_{EY}$ is not convex in $W$ jointly, so fitting is only
  guaranteed to reach a stationary point, and different `random_state` initialisations can land
  on different ones — the same caveat that already applies to `CCAEY`'s gradient descent.
- No optional dependency is required: all three are built entirely on numpy.
