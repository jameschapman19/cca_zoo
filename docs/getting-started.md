# Getting Started

## Installation

Install the core package with pip:

```bash
pip install cca-zoo
```

The core package requires Python ≥ 3.10 and provides all linear and nonparametric methods.

### Optional extras

Install extras for additional method families:

```bash
pip install cca-zoo[deep]          # DCCA variants (PyTorch + Lightning)
pip install cca-zoo[probabilistic] # Probabilistic CCA (NumPyro + JAX)
pip install cca-zoo[tree]          # TreeCCA (XGBoost + LightGBM)
pip install cca-zoo[all]           # Everything above
```

!!! note "PyTorch installation"
    For GPU support or specific CUDA versions, follow the
    [PyTorch installation guide](https://pytorch.org/get-started/locally/) before
    installing `cca-zoo[deep]`.

---

## Core concepts

### Views

CCA-Zoo expects data as a **list of arrays**, one per view:

```python
views = [X1, X2]  # two views
views = [X1, X2, X3]  # three views
```

Each array has shape `(n_samples, n_features_i)`. All views must share the same number of rows.

### The estimator API

Every model follows the same three-step pattern:

```python
from cca_zoo.linear import CCA

model = CCA(latent_dimensions=2)  # 1. construct
model.fit(views)  # 2. fit
z = model.transform(views)  # 3. use
```

Or equivalently:

```python
z = CCA(latent_dimensions=2).fit_transform(views)
```

### Evaluating fit quality

`score` returns the average pairwise canonical correlation per latent dimension:

```python
corrs = model.score(views)  # np.ndarray, shape (latent_dimensions,)
print(corrs)  # e.g. [0.94, 0.87]
```

### Inspecting weights

After fitting, `model.weights` is a list of weight matrices (one per view):

```python
W1, W2 = model.weights  # each shape (n_features_i, latent_dimensions)
```

---

## Quick start

### Two-view CCA

```python
import numpy as np
from cca_zoo.datasets import JointData
from cca_zoo.linear import CCA

# Simulate data from a linear latent variable model
data = JointData(
    n_views=2,
    n_samples=200,
    n_features=[50, 50],
    latent_dimensions=2,
    signal_to_noise=2.0,
    random_state=0,
)
train_views = data.sample()
test_views = data.sample()

# Fit and evaluate
model = CCA(latent_dimensions=2).fit(train_views)
print("Canonical correlations:", model.score(test_views))

# Project into the shared latent space
z1, z2 = model.transform(test_views)
print("Latent shape:", z1.shape)  # (200, 2)
```

### Multiview CCA (≥2 views)

```python
from cca_zoo.linear import MCCA

data = JointData(n_views=3, n_samples=200, n_features=30, random_state=0)
views = data.sample()

model = MCCA(latent_dimensions=2).fit(views)
print(model.score(views))
```

### Regularised CCA

```python
from cca_zoo.linear import rCCA

# c controls the ridge penalty (0 = CCA, 1 = PLS)
model = rCCA(latent_dimensions=2, c=0.1).fit(train_views)
```

### Kernel CCA

```python
from cca_zoo.nonparametric import KCCA

model = KCCA(latent_dimensions=2, kernel="rbf", gamma=0.01, c=0.1).fit(train_views)
z1, z2 = model.transform(test_views)
```

### Hyperparameter search

CCA-Zoo's `GridSearchCV` wraps sklearn's grid search with a multiview interface:

```python
from cca_zoo.model_selection import GridSearchCV
from cca_zoo.nonparametric import KCCA

param_grid = {"c": [0.01, 0.1, 1.0], "gamma": [0.01, 0.1]}
gs = GridSearchCV(KCCA(latent_dimensions=2, kernel="rbf"), param_grid, cv=5)
gs.fit(train_views)
print("Best params:", gs.best_params_)
```

---

## Next steps

- **[User Guide — Linear Methods](user-guide/linear.md)** — learn which linear model to choose
- **[User Guide — Nonparametric Methods](user-guide/nonparametric.md)** — kernel CCA explained
- **[User Guide — Tree Methods](user-guide/tree.md)** — gradient-boosted-tree CCA
- **[User Guide — Deep Methods](user-guide/deep.md)** — using neural network encoders
- **[API Reference](api/linear.md)** — full class and method documentation
