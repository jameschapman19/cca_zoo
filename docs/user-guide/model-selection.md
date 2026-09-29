# Model Selection

Every `cca_zoo` model is a `sklearn.base.BaseEstimator`, but its `fit`/`transform`/`score`
take a `list[ArrayLike]` of per-view arrays rather than a single 2-D `X`. That's the one
thing that stops sklearn's own model-selection tools (`GridSearchCV`, `cross_val_score`,
`Pipeline`, ...) from working with it directly — they need to slice `X` by row to build
folds, and can't do that across a Python list of differently-shaped arrays.

`cca_zoo.model_selection` closes that gap. Its searches and cross-validation functions
are sklearn's, taking a list of views: they stack the views into one array for sklearn and
split them back for the model, so folds, scores and results behave exactly as in sklearn.
sklearn's `Pipeline` needs no adapter; `cca_zoo.preprocessing.PerViewTransformer` applies a
transformer to each view within it.

---

## GridSearchCV

`GridSearchCV` finds the hyperparameters that maximise the average canonical correlation on
held-out folds, using sklearn's cross-validation machinery under the hood.

```python
from cca_zoo.model_selection import GridSearchCV
from cca_zoo.linear import RidgeCCA

param_grid = {"shrinkage": [0.001, 0.01, 0.1, 1.0]}
gs = GridSearchCV(RidgeCCA(n_components=2), param_grid=param_grid, cv=5)
gs.fit([X1, X2])

print("Best shrinkage:", gs.best_params_["shrinkage"])
print("Best CV score:", gs.best_score_)

# Use the refitted best model directly
best_model = gs.best_estimator_
z1, z2 = best_model.transform([X1, X2])
```

### Per-view parameters

Many CCA models accept per-view parameters as a scalar (broadcast to all views) or an
explicit list, e.g. `KCCA(shrinkage=[0.01, 0.1])`. To search each view's value independently,
suffix the parameter name with `__<view index>`; sklearn then searches the full Cartesian
product across views:

```python
from cca_zoo.model_selection import GridSearchCV
from cca_zoo.nonparametric import KCCA

param_grid = {
    "shrinkage__0": [0.01, 0.1, 1.0],
    "shrinkage__1": [0.001, 0.01],
}  # all 6 combinations

gs = GridSearchCV(
    KCCA(n_components=2, kernel="rbf", gamma=0.01), param_grid=param_grid, cv=5
)
gs.fit([X1, X2])
print(gs.best_params_)  # e.g. {"shrinkage__0": 0.1, "shrinkage__1": 0.01}
print(gs.best_estimator_.shrinkage)  # [0.1, 0.01]
```

An index you don't mention in the grid keeps the estimator's current value for that view,
so `param_grid={"shrinkage__0": [...]}` alone only tunes view 0, leaving view 1 fixed.

### Accessing results

`GridSearchCV` exposes the standard sklearn attributes:

```python
import pandas as pd

# Full CV results table
df = pd.DataFrame(gs.cv_results_)
print(
    df[
        [
            "param_shrinkage__0",
            "param_shrinkage__1",
            "mean_test_score",
            "std_test_score",
        ]
    ].sort_values("mean_test_score")
)

# Best parameters and score
print(gs.best_params_)
print(gs.best_score_)

# Best estimator (already refitted on the full training set)
best = gs.best_estimator_
```

### Custom refit rules

As in sklearn, `refit` also takes a callable that receives `cv_results_` and returns the index of
the candidate to refit. When CV scores are noisy, the top mean score tends to favour complex
candidates, and a common remedy is the one-standard-error rule (Breiman's CART, `glmnet`'s
`lambda.1se`): take the simplest candidate whose mean score is within one standard error of the
best. sklearn's [Balance model complexity and cross-validated
score](https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_refit_callable.html)
example shows the pattern; for `MARSCCA`'s `nprune` (smaller is simpler):

```python
import numpy as np
from cca_zoo.gam import MARSCCA
from cca_zoo.model_selection import GridSearchCV


def one_standard_error(cv_results):
    mean = cv_results["mean_test_score"]
    se = cv_results["std_test_score"] / np.sqrt(5)  # 5 CV splits
    best = np.argmax(mean)
    eligible = np.flatnonzero(mean >= mean[best] - se[best])
    return eligible[np.argmin(cv_results["param_nprune"][eligible])]


gs = GridSearchCV(
    MARSCCA(degree=2, nk=40),
    {"nprune": [2, 4, 8, 12, 16, 24, 32, 48, 80]},
    cv=5,
    refit=one_standard_error,
).fit([X1, X2])
```

`cv_results_` carries the same parameter names you passed in the grid (`param_nprune`, not the
internal `param_estimator__nprune`). The successive-halving searches choose their final
candidate themselves and, like sklearn's, take only `refit=True`/`False`.

---

## RandomizedSearchCV

For a continuous hyperparameter like `shrinkage`, sampling a distribution is usually more efficient
than searching a fixed grid. `RandomizedSearchCV` mirrors
`sklearn.model_selection.RandomizedSearchCV`: pass a distribution (anything with an `rvs`
method, e.g. `scipy.stats.loguniform`) instead of a list of values, and it draws `n_iter`
random parameter settings rather than trying every grid point.

```python
from scipy.stats import loguniform
from cca_zoo.model_selection import RandomizedSearchCV
from cca_zoo.linear import RidgeCCA

rs = RandomizedSearchCV(
    RidgeCCA(n_components=2),
    param_distributions={"shrinkage": loguniform(1e-4, 1.0)},
    n_iter=20,
    cv=5,
    random_state=0,
)
rs.fit([X1, X2])
print("Best shrinkage:", rs.best_params_["shrinkage"])
```

`HalvingGridSearchCV` and `HalvingRandomSearchCV` are sklearn's successive-halving searches
in the same way. Each search class extends its sklearn namesake, so it takes exactly that
class's parameters and follows it as sklearn changes; only `fit`, `transform` and `score`
take a list of views, and parameter names are the model's own.

### OptunaSearchCV

With the `optuna` extra (`pip install 'cca-zoo[optuna]'`), `OptunaSearchCV` is
`optuna_integration.OptunaSearchCV` on a list of views, with Optuna's samplers, pruners and
parameters:

```python
from optuna.distributions import FloatDistribution
from cca_zoo.model_selection import OptunaSearchCV

search = OptunaSearchCV(
    RidgeCCA(n_components=2),
    {"shrinkage__0": FloatDistribution(0, 1), "shrinkage__1": FloatDistribution(0, 1)},
    n_trials=50,
    cv=5,
).fit([X1, X2])
print(search.best_params_)
```

Optuna's own records, such as `search.study_.trials_dataframe()`, show the parameters with an
internal `estimator__` prefix.

---

## Cross-validation

`cross_val_score`, `cross_validate`, `cross_val_predict`, `learning_curve` and
`validation_curve` are sklearn's functions taking a list of views; every other argument is
passed to sklearn. `validation_curve` accepts per-view names such as `"shrinkage__0"`.
`cross_val_predict` returns each view's out-of-fold scores, from which out-of-sample
canonical correlations follow. Each fold's model fixes its own signs, so compare views
within a component rather than scores across folds.

```python
from cca_zoo.linear import CCA, RidgeCCA
from cca_zoo.metrics import pairwise_correlations
from cca_zoo.model_selection import (
    cross_val_predict,
    cross_val_score,
    cross_validate,
    validation_curve,
)

scores = cross_val_score(CCA(n_components=2), [X1, X2], cv=5)
results = cross_validate(CCA(n_components=2), [X1, X2], cv=5, return_train_score=True)
train, test = validation_curve(
    RidgeCCA(), [X1, X2], "shrinkage__0", [0.0, 0.1, 1.0], cv=5
)
out_of_fold = pairwise_correlations(
    cross_val_predict(CCA(n_components=2), [X1, X2], cv=5)
)
```

---

## Pipelines

A `Pipeline` of `PerViewTransformer` steps and a model takes a list of views, so preprocessing
is refitted within each fold. Per-view names reach through it as usual:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from cca_zoo.linear import RidgeCCA
from cca_zoo.model_selection import GridSearchCV
from cca_zoo.preprocessing import PerViewTransformer

pipeline = Pipeline(
    [("scale", PerViewTransformer(StandardScaler())), ("cca", RidgeCCA())]
)
gs = GridSearchCV(pipeline, {"cca__shrinkage__0": [0.0, 0.1, 1.0]}, cv=5).fit([X1, X2])
```

## Custom scoring

The default score is the model's `score`, the mean canonical correlation. Any search or
cross-validation function takes a callable `scoring(estimator, views)` instead, or a dict
of them, called with the fitted model and the held-out views:

```python
from cca_zoo.metrics import pairwise_correlations


def first_correlation(estimator, views):
    return pairwise_correlations(estimator.transform(views))[0, 1, 0]


scores = cross_val_score(CCA(n_components=2), [X1, X2], cv=5, scoring=first_correlation)
```

---

## Full example: tuning kernel CCA

```python
from cca_zoo.datasets import make_joint_data
from cca_zoo.model_selection import GridSearchCV
from cca_zoo.nonparametric import KCCA

# Simulate data
views = make_joint_data(
    n_samples=200,
    n_features=[30, 30],
    n_components=2,
    signal_to_noise=2.0,
    random_state=0,
)

# Grid search over kernel and regularisation
param_grid = {
    "kernel": ["rbf", "poly"],
    "shrinkage": [0.01, 0.1, 1.0],
    "gamma": [0.01, 0.1],
}
gs = GridSearchCV(
    KCCA(n_components=2),
    param_grid=param_grid,
    cv=5,
)
gs.fit(views)

print("Best params:", gs.best_params_)
print("Best score: ", gs.best_score_)
```

---

## Tips

- `score` is the mean canonical correlation across all `n_components`, averaged over
  all pairwise view combinations.
- Cross-validation is done on the full set of views passed to `fit`; train/test splits are
  row-wise (same rows held out across all views).
- For sparse CCA methods, tune `alpha`, `l1_bound` or `span` just like any other
  hyperparameter.
- When the grid is large, prefer `RandomizedSearchCV`, or a coarse-to-fine `GridSearchCV`:
  search a coarse grid first, then refine around the best value.

---

## Assessing significance

`permutation_test_significance` answers two different questions: is a canonical
correlation stronger than you'd expect by chance, and which individual features
reliably drive it?

```python
from cca_zoo.linear import CCA
from cca_zoo.model_selection import permutation_test_significance

result = permutation_test_significance(
    CCA(n_components=2), [X1, X2], n_permutations=1000, random_state=0
)

print("Canonical correlations:", result.correlations)
print("p-values (per dimension):", result.p_values)

# Per-feature, per-dimension p-values for view 1's loadings
print(result.loading_p_values[0])
```

It works by refitting the model many times on data where every view except the first
has had its rows independently shuffled, breaking the true cross-view relationship while
keeping each view's own covariance structure intact. `p_values` compares each
dimension's true correlation directly against its shuffled counterparts. For the
loadings, a shuffled refit isn't guaranteed to recover components in the same order or
sign as the true fit — permutation can rotate or reflect near-tied dimensions — so each
permutation's loadings are first realigned to the true fit (`scipy.linalg.orthogonal_procrustes`)
before being compared feature-by-feature. This follows the resampling-based significance
testing approach used in the neuroimaging CCA/PLS literature (Xia et al. 2018; McIntosh &
Lobaugh 2004).

`n_permutations` trades off precision against runtime (each permutation refits the model
from scratch); pass `n_jobs` to parallelise across permutations.
