# Model Selection

Every `cca_zoo` model is a `sklearn.base.BaseEstimator`, but its `fit`/`transform`/`score`
take a `list[ArrayLike]` of per-view arrays rather than a single 2-D `X`. That's the one
thing that stops sklearn's own model-selection tools (`GridSearchCV`, `cross_val_score`,
`Pipeline`, ...) from working with it directly — they need to slice `X` by row to build
folds, and can't do that across a Python list of differently-shaped arrays.

`cca_zoo.model_selection.MultiviewWrapper` closes that gap: it horizontally stacks the
views into one array on the way in, and splits them back before calling the wrapped
estimator. Once wrapped, the estimator is an ordinary sklearn estimator, so *any* sklearn
tool — not just grid search — applies unmodified. `GridSearchCV` and `RandomizedSearchCV`
below do this wrapping for you and otherwise delegate entirely to
`sklearn.model_selection.GridSearchCV` / `RandomizedSearchCV`.

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

Many CCA models accept per-view parameters as a scalar (broadcast to all views) or an
explicit list, e.g. `KCCA(c=[0.01, 0.1])`. To search each view's value independently,
suffix the parameter name with `__<view index>`; sklearn then searches the full Cartesian
product across views:

```python
from cca_zoo.model_selection import GridSearchCV
from cca_zoo.nonparametric import KCCA

param_grid = {"c__0": [0.01, 0.1, 1.0], "c__1": [0.001, 0.01]}  # all 6 combinations

gs = GridSearchCV(
    KCCA(latent_dimensions=2, kernel="rbf", gamma=0.01), param_grid=param_grid, cv=5
)
gs.fit([X1, X2])
print(gs.best_params_)  # e.g. {"c__0": 0.1, "c__1": 0.01}
print(gs.best_estimator_.c)  # [0.1, 0.01]
```

An index you don't mention in the grid keeps the estimator's current value for that view,
so `param_grid={"c__0": [...]}` alone only tunes view 0, leaving view 1 fixed.

### Accessing results

`GridSearchCV` exposes the standard sklearn attributes:

```python
import pandas as pd

# Full CV results table
df = pd.DataFrame(gs.cv_results_)
print(
    df[["param_c", "mean_test_score", "std_test_score"]].sort_values("mean_test_score")
)

# Best parameters and score
print(gs.best_params_)
print(gs.best_score_)

# Best estimator (already refitted on the full training set)
best = gs.best_estimator_
```

---

## RandomizedSearchCV

For a continuous hyperparameter like `c`, sampling a distribution is usually more efficient
than searching a fixed grid. `RandomizedSearchCV` mirrors
`sklearn.model_selection.RandomizedSearchCV`: pass a distribution (anything with an `rvs`
method, e.g. `scipy.stats.loguniform`) instead of a list of values, and it draws `n_iter`
random parameter settings rather than trying every grid point.

```python
from scipy.stats import loguniform
from cca_zoo.model_selection import RandomizedSearchCV
from cca_zoo.linear import rCCA

rs = RandomizedSearchCV(
    rCCA(latent_dimensions=2),
    param_distributions={"c": loguniform(1e-4, 1.0)},
    n_iter=20,
    cv=5,
    random_state=0,
)
rs.fit([X1, X2])
print("Best c:", rs.best_params_["c"])
```

---

## Using sklearn tools directly

`GridSearchCV` and `RandomizedSearchCV` cover the common case, but they don't need to be
the only way in: `MultiviewWrapper` is the same adapter they use internally, and it's
public, so any other sklearn model-selection tool — `HalvingGridSearchCV`,
`cross_val_score`, `cross_validate`, `learning_curve`, `Pipeline`, ... — works with it too.

```python
import numpy as np
from sklearn.model_selection import cross_validate
from cca_zoo.model_selection import MultiviewWrapper
from cca_zoo.linear import CCA

views = [X1, X2]
split_indices = [v.shape[1] for v in views]
wrapper = MultiviewWrapper(CCA(latent_dimensions=2), split_indices=split_indices)

results = cross_validate(wrapper, np.hstack(views), cv=5, return_train_score=True)
print(results["test_score"])
```

---

## Full example: tuning kernel CCA

```python
import numpy as np
from cca_zoo.datasets import JointData
from cca_zoo.model_selection import GridSearchCV
from cca_zoo.nonparametric import KCCA

# Simulate data
data = JointData(
    n_views=2,
    n_samples=200,
    n_features=[30, 30],
    latent_dimensions=2,
    signal_to_noise=2.0,
    random_state=0,
)
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
- When the grid is large, prefer `RandomizedSearchCV`, or a coarse-to-fine `GridSearchCV`:
  search a coarse grid first, then refine around the best value.
- `MultiviewWrapper` composes with any sklearn model-selection tool, not just the two
  classes above — reach for it directly when you need `HalvingGridSearchCV`,
  `cross_val_score`, or a `Pipeline` step.

---

## Assessing significance

`permutation_test_significance` answers two different questions: is a canonical
correlation stronger than you'd expect by chance, and which individual features
reliably drive it?

```python
from cca_zoo.linear import CCA
from cca_zoo.model_selection import permutation_test_significance

result = permutation_test_significance(
    CCA(latent_dimensions=2), [X1, X2], n_permutations=1000, random_state=0
)

print("Canonical correlations:", result.correlations_)
print("p-values (per dimension):", result.p_values_)

# Per-feature, per-dimension p-values for view 1's loadings
print(result.loading_p_values_[0])
```

It works by refitting the model many times on data where every view except the first
has had its rows independently shuffled, breaking the true cross-view relationship while
keeping each view's own covariance structure intact. `p_values_` compares each
dimension's true correlation directly against its shuffled counterparts. For the
loadings, a shuffled refit isn't guaranteed to recover components in the same order or
sign as the true fit — permutation can rotate or reflect near-tied dimensions — so each
permutation's loadings are first realigned to the true fit via `procrustes_rotation`
before being compared feature-by-feature. This follows the resampling-based significance
testing approach used in the neuroimaging CCA/PLS literature (Xia et al. 2018; McIntosh &
Lobaugh 2004).

`n_permutations` trades off precision against runtime (each permutation refits the model
from scratch); pass `n_jobs` to parallelise across permutations.
