# Tree Methods

The `cca_zoo.tree` module provides `XGBoostCCA` and `LightGBMCCA`, nonlinear multiview CCA
methods that use gradient-boosted trees as the per-view encoders, via
[XGBoost](https://xgboost.readthedocs.io/) or [LightGBM](https://lightgbm.readthedocs.io/)
respectively. Both share the same `TreeCCA` base class and fitting recipe, differing only in
which gradient-boosting library trains the per-view encoders. Install the module with:

```bash
pip install cca-zoo[tree]
```

---

## Background

`TreeCCA` maximises the same unconstrained Eckart-Young (EY) objective used by the stochastic
`*_EY` models in `cca_zoo.linear` and by `DCCAEY` in `cca_zoo.deep` (the numpy-based models
share the exact same implementation, in `cca_zoo._utils._ey`):

$$
\mathcal{L}_{EY} = -2 \operatorname{tr}(C) + \operatorname{tr}(V V)
$$

where, for embeddings $Z_i = f_i(X_i)$, $C$ is the mean pairwise cross-covariance (including
$i = j$ terms) and $V$ the mean auto-covariance across all views. `TreeCCA` uses a
gradient-boosted-tree ensemble in place of a linear map or neural network as the function class
for each $f_i$.

Each of the `latent_dimensions` canonical components is a separate scalar booster per view.
Training proceeds by alternating (Gauss-Seidel) gradient boosting: each round, for every view in
turn, the EY-loss gradient is computed from the current embeddings and used as a custom
regression objective to add one tree to that view's boosters; with `gauss_seidel=True` (the
default) the gradient is recomputed from the freshest embeddings before moving to the next view.
Boosters start from a random-orthogonal, unit-variance initial embedding per view — random
directions in feature space, rescaled so each component has unit variance. Unit-variance scaling
is what matters for a well-conditioned, non-vanishing gradient at round zero; orthogonality keeps
the initial components uncorrelated.

Because each component is its own boosted-tree ensemble, per-component feature importance (split
gain) is available directly from the fitted boosters — no separate interpretability method (such
as SHAP) is required.

**When to use:** Nonlinear multiview CCA where you want built-in, per-component feature
importance, and where a tree-based model is a natural fit for the data (e.g. tabular data with
mixed scales, non-smooth or threshold-like relationships).

---

## Basic usage

`XGBoostCCA` and `LightGBMCCA` are separate classes rather than one class with a `backend=`
switch, so `TreeCCA` itself cannot be instantiated directly — pick the concrete class you want:

```python
from cca_zoo.tree import XGBoostCCA

model = XGBoostCCA(latent_dimensions=2, n_estimators=200, max_depth=5).fit([X1, X2])
z1, z2 = model.transform([X1, X2])
corrs = model.score([X1, X2])

# XGBoostCCA also supports more than two views
model3 = XGBoostCCA(latent_dimensions=2, n_estimators=200).fit([X1, X2, X3])
```

Use `LightGBMCCA` to train with LightGBM instead (requires `pip install lightgbm`, included in
the `tree` extra):

```python
from cca_zoo.tree import LightGBMCCA

model = LightGBMCCA(latent_dimensions=2).fit([X1, X2])
```

## Feature importance

Neither class has linear weight matrices, so `model.weights` raises `NotImplementedError`. Use
the fitted `boosters_` attribute instead — a `list[list[Booster]]` indexed `[view][component]`:

```python
model = XGBoostCCA(latent_dimensions=2).fit([X1, X2])

# Split-gain feature importance for view 1, canonical component 0
importance = model.boosters_[0][0].get_score(importance_type="gain")

# Equivalent for LightGBMCCA
# importance = model.boosters_[0][0].feature_importance(importance_type="gain")
```

---

## Key parameters

| Parameter | Description |
|---|---|
| `n_estimators` | Boosting rounds (trees added per booster). Higher values fit more complex relationships but risk overfitting and cost more time. |
| `max_depth` | Maximum tree depth. |
| `learning_rate` | Boosting shrinkage. |
| `subsample`, `colsample_bytree` | Row/column subsampling ratios per tree, for regularisation. |
| `gauss_seidel` | Use freshly-updated view-1 embeddings when computing view 2's gradient each round (default `True`); set `False` for Jacobi-style stale updates. |
| `random_state` | Seed for the boosters and the random-orthogonal initial embedding. |

Hyperparameters are best selected by cross-validation with `GridSearchCV` from
`cca_zoo.model_selection`, as for other models.

---

## Practical notes

- `XGBoostCCA` and `LightGBMCCA` both support 2 or more views.
- `latent_dimensions` must not exceed the number of features in any view (the random-orthogonal
  initialisation draws that many orthogonal directions in feature space).
- Unlike `KCCA`, neither class stores the training data for inference — new data is passed
  directly through the fitted boosters, so `transform` on held-out data is inexpensive.
- Reference: Chapman, J. (2026). *TreeCCA: Canonical Correlation Analysis via
  Gradient-Boosted Trees.* arXiv:2607.27027.
