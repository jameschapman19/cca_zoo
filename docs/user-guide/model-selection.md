# Model Selection

`cca_zoo.model_selection` provides cross-validated hyperparameter search for multiview models.
It wraps sklearn's `GridSearchCV` with a multiview-compatible interface.

---

## GridSearchCV

`GridSearchCV` finds the hyperparameters that maximise the average canonical correlation on
held-out folds, using sklearn's cross-validation machinery under the hood.

```python
from cca_zoo.model_selection import GridSearchCV
from cca_zoo.linear import rCCA

param_grid = {"c": [0.001, 0.01, 0.1, 1.0]}
gs = GridSearchCV(rCCA(latent_dimensions=2), param_grid=param_grid, cv=5)
gs.fit([X1, X2])

print("Best c:", gs.best_params_["c"])
print("Best CV score:", gs.best_score_)

# Use the refitted best model directly
best_model = gs.best_estimator_
z1, z2 = best_model.transform([X1, X2])
```

### Per-view parameters

Many CCA models accept per-view parameters as a scalar (broadcast to all views) or a list.
In the parameter grid, use lists to specify per-view values:

```python
from cca_zoo.model_selection import GridSearchCV
from cca_zoo.nonparametric import KCCA

# Scalar c applies to all views
param_grid = {"c": [0.01, 0.1, 1.0]}

# Per-view c (list of two values for a two-view model)
param_grid = {"c": [[0.01, 0.1], [0.1, 1.0]]}

gs = GridSearchCV(KCCA(latent_dimensions=2, kernel="rbf", gamma=0.01),
                  param_grid=param_grid, cv=5)
gs.fit([X1, X2])
```

### Accessing results

`GridSearchCV` exposes the standard sklearn attributes:

```python
import pandas as pd

# Full CV results table
df = pd.DataFrame(gs.cv_results_)
print(df[["param_c", "mean_test_score", "std_test_score"]].sort_values("mean_test_score"))

# Best parameters and score
print(gs.best_params_)
print(gs.best_score_)

# Best estimator (already refitted on the full training set)
best = gs.best_estimator_
```

---

## Full example: tuning kernel CCA

```python
import numpy as np
from cca_zoo.datasets import JointData
from cca_zoo.model_selection import GridSearchCV
from cca_zoo.nonparametric import KCCA

# Simulate data
data = JointData(n_views=2, n_samples=200, n_features=[30, 30],
                 latent_dimensions=2, signal_to_noise=2.0, random_state=0)
views = data.sample()

# Grid search over kernel and regularisation
param_grid = {
    "kernel": ["rbf", "poly"],
    "c": [0.01, 0.1, 1.0],
    "gamma": [0.01, 0.1],
}
gs = GridSearchCV(
    KCCA(latent_dimensions=2),
    param_grid=param_grid,
    cv=5,
)
gs.fit(views)

print("Best params:", gs.best_params_)
print("Best score: ", gs.best_score_)
```

---

## Tips

- `score` is the mean canonical correlation across all `latent_dimensions`, averaged over
  all pairwise view combinations.
- Cross-validation is done on the full set of views passed to `fit`; train/test splits are
  row-wise (same rows held out across all views).
- For sparse CCA methods, tune `tau` or `alpha` just like any other hyperparameter.
- When the grid is large, prefer a coarse-to-fine search: run `GridSearchCV` on a coarse grid
  first, then refine around the best value.
