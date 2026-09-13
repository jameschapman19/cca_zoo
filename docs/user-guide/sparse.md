# Sparse Methods

The `cca_zoo.sparse` module provides `ElasticNetCCA`, a sparse linear multiview CCA method that
learns per-view canonical weight vectors under an elastic-net (L1 + L2) penalty. It has no
optional dependency: it is built entirely on numpy, already required by `cca_zoo`.

---

## Background

`ElasticNetCCA` minimises the same unconstrained Eckart-Young (EY) objective used by the
stochastic `*_EY` models in `cca_zoo.linear`, by `DCCAEY` in `cca_zoo.deep`, by `TreeCCA` in
`cca_zoo.tree`, and by `GAMCCA`/`GaussianProcessCCA` (all share the exact same implementation, in
`cca_zoo._utils._ey`), with an elastic-net penalty added on the per-view weights:

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

---

## Basic usage

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

| Parameter | Description |
|---|---|
| `alpha` | Overall elastic-net penalty strength. |
| `l1_ratio` | Elastic-net mixing parameter in `[0, 1]`; 0 is pure ridge, 1 is pure lasso. |
| `max_iter` | Maximum number of full coordinate-descent sweeps (every view, feature, and component once each). |
| `tol` | Convergence tolerance on the penalised objective's change between consecutive sweeps. |
| `random_state` | Seed for the initial weights. |

Hyperparameters are best selected by cross-validation with `GridSearchCV` from
`cca_zoo.model_selection`, as for other models.

---

## Practical notes

- `ElasticNetCCA` supports 2 or more views.
- `latent_dimensions` must not exceed the number of features in any view.
- Like every EY-loss model, $\mathcal{L}_{EY}$ is not convex in $W$ jointly, so coordinate
  descent is only guaranteed to reach a stationary point, and different `random_state`
  initialisations can land on different ones — the same caveat that already applies to
  `CCAEY`'s gradient descent.
- No optional dependency is required: `ElasticNetCCA` is built entirely on numpy.
