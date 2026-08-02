# Tree Methods

The `cca_zoo.tree` module provides `TreeCCA`, a nonlinear two-view CCA method that uses
gradient-boosted trees (via [XGBoost](https://xgboost.readthedocs.io/)) as the per-view encoders.
Install it with:

```bash
pip install cca-zoo[tree]
```

---

## Background

`TreeCCA` maximises the Eckart-Young (EY) unconstrained-CCA objective

$$
\mathcal{L}_{EY}(Z_1, Z_2) = 2 \operatorname{tr}(C_{12}) - \operatorname{tr}(V_{11} + V_{22})
$$

where $Z_i = f_i(X_i)$ are the encoded views, $C_{12}$ is their cross-covariance, and $V_{ii}$
their within-view covariance. This is the same objective used by the stochastic gradient-descent
`*_EY` models in `cca_zoo.linear` and by `DCCA_EY` in `cca_zoo.deep` — `TreeCCA` uses a
gradient-boosted-tree ensemble in place of a linear map or neural network as the function class
for $f_i$.

Each of the `latent_dimensions` canonical components is a separate scalar XGBoost booster per
view. Training proceeds by alternating (Gauss-Seidel) gradient boosting: each round, the EY-loss
gradient with respect to the current embedding is computed and used as a custom regression
objective to add one tree to every booster.

Because each component is its own boosted-tree ensemble, per-component feature importance (split
gain) is available directly from the fitted boosters — no separate interpretability method (such
as SHAP) is required.

**When to use:** Nonlinear two-view CCA where you want built-in, per-component feature
importance, and where a tree-based model is a natural fit for the data (e.g. tabular data with
mixed scales, non-smooth or threshold-like relationships).

---

## Basic usage

```python
from cca_zoo.tree import TreeCCA

model = TreeCCA(latent_dimensions=2, n_estimators=200, max_depth=5).fit([X1, X2])
z1, z2 = model.transform([X1, X2])
corrs = model.score([X1, X2])
```

## Feature importance

`TreeCCA` has no linear weight matrices, so `model.weights` raises `NotImplementedError`. Use the
fitted `boosters_` attribute instead — a `list[list[xgboost.Booster]]` indexed `[view][component]`:

```python
model = TreeCCA(latent_dimensions=2).fit([X1, X2])

# Split-gain feature importance for view 1, canonical component 0
importance = model.boosters_[0][0].get_score(importance_type="gain")
```

---

## Key parameters

| Parameter | Description |
|---|---|
| `n_estimators` | Boosting rounds (trees added per booster). Higher values fit more complex relationships but risk overfitting and cost more time. |
| `max_depth` | Maximum tree depth. |
| `learning_rate` | Boosting shrinkage (XGBoost `eta`). |
| `subsample`, `colsample_bytree` | Row/column subsampling ratios per tree, for regularisation. |
| `gauss_seidel` | Use freshly-updated view-1 embeddings when computing view 2's gradient each round (default `True`); set `False` for Jacobi-style stale updates. |

Hyperparameters are best selected by cross-validation with `GridSearchCV` from
`cca_zoo.model_selection`, as for other models.

---

## Practical notes

- `TreeCCA` currently supports exactly two views.
- Unlike `KCCA`, `TreeCCA` does not store the training data for inference — new data is passed
  directly through the fitted boosters, so `transform` on held-out data is inexpensive.
- Reference: Chapman, J., Wells, L., & Lawry Aguila, L. (2024). *Unconstrained stochastic CCA:
  Unifying multiview and self-supervised learning.* arXiv:2310.01012.
