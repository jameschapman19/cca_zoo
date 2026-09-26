# Nonparametric Methods

The `cca_zoo.nonparametric` module provides kernel- and graph-based CCA methods that can capture
nonlinear relationships between views without explicitly constructing feature maps.

---

## Background

Kernel methods replace the inner product $\mathbf{x}_i^\top \mathbf{x}_j$ with a kernel
function $k(\mathbf{x}_i, \mathbf{x}_j)$, implicitly mapping data into a potentially
infinite-dimensional reproducing kernel Hilbert space (RKHS). The canonical directions are found
in that RKHS and expressed via the *dual* (kernel coefficient) representation.

**Supported kernels** (passed directly to `sklearn.metrics.pairwise_kernels`):

| Kernel | String | Key params |
|---|---|---|
| Linear | `"linear"` | — |
| Polynomial | `"poly"` | `degree`, `coef0` |
| RBF / Gaussian | `"rbf"` | `gamma` |
| Sigmoid | `"sigmoid"` | `gamma`, `coef0` |
| Custom callable | any `Callable` | via `kernel_params` |

---

## KCCA — Kernel CCA

**When to use:** Nonlinear two-view CCA. The kernel generalises MCCA.

KCCA finds dual variables $\boldsymbol{\alpha}_i$ (one set per view) by solving:

$$
A \boldsymbol{\alpha} = \lambda B \boldsymbol{\alpha}
$$

where:

- $A$ is the block off-diagonal kernel cross-covariance matrix
- $B = \mathrm{block\_diag}(c_i K_i + (1-c_i) K_i^2)$ regularises the within-view kernel matrices

The parameter `c` controls regularisation: larger `c` → stronger regularisation.

```python
from cca_zoo.nonparametric import KCCA

# Linear kernel (recovers classical CCA in feature space)
model = KCCA(latent_dimensions=2, kernel="linear", c=0.1).fit([X1, X2])

# RBF kernel
model = KCCA(latent_dimensions=2, kernel="rbf", gamma=0.01, c=0.1).fit([X1, X2])

# Polynomial kernel
model = KCCA(latent_dimensions=2, kernel="poly", degree=3, c=0.1).fit([X1, X2])

# Per-view kernel parameters (list = one entry per view)
model = KCCA(
    latent_dimensions=2,
    kernel=["rbf", "poly"],
    gamma=[0.01, None],
    degree=[1, 3],
    c=[0.1, 0.5],
).fit([X1, X2])
```

### Transform

At test time, KCCA computes kernel matrices between test and training points and projects via
the fitted dual variables:

```python
z1, z2 = model.transform([X1_test, X2_test])
```

---

## KGCCA — Kernel Generalised CCA

**When to use:** Nonlinear extension of GCCA for three or more views.

KGCCA builds a shared kernel-space latent representation analogously to GCCA:

$$
Q = \sum_i \mu_i K_i \, B_i^{-1} \, K_i
$$

where $B_i = c_i K_i + (1-c_i) K_i^2$.

```python
from cca_zoo.nonparametric import KGCCA

model = KGCCA(latent_dimensions=2, kernel="rbf", gamma=0.01, c=0.1).fit([X1, X2, X3])
```

---

## KTCCA — Kernel Tensor CCA

**When to use:** Captures higher-order correlations in the kernel space for three or more views.

KTCCA whitens the kernel matrices and builds a cross-moment tensor in the RKHS, then applies
PARAFAC decomposition:

```python
from cca_zoo.nonparametric import KTCCA

model = KTCCA(latent_dimensions=2, kernel="rbf", gamma=0.01, c=0.1, random_state=0).fit(
    [X1, X2, X3]
)
```

---

## ManifoldCCA — transductive manifold CCA

**When to use:** Two views share a single underlying coordinate that is embedded
*nonlinearly and differently* in each view's raw features (e.g. two different curved/spiral
parameterisations), so no linear map and no single global kernel connects the two ambient
spaces well, but each view's own local (k-nearest-neighbour) structure still respects the
shared ordering.

Unlike `KCCA`, which replaces the inner product with a global kernel, `ManifoldCCA` replaces the
within-view *covariance* with a graph operator built from that view's own local neighbourhood
structure -- the normalised graph Laplacian ($M = I - D^{-1/2}WD^{-1/2}$, `method="laplacian"`,
matching `sklearn.manifold.SpectralEmbedding`) or the LLE reconstruction operator
($M = (I-W)^\top(I-W)$, `method="lle"`). Since there's no feature map at all here, the "weight"
the joint eigenproblem solves for *is* each view's training-set embedding directly:

```python
from cca_zoo.nonparametric import ManifoldCCA

model = ManifoldCCA(method="laplacian", n_neighbors=10, latent_dimensions=1).fit(
    [X1, X2]
)
train_embedding = model.embedding_  # (n_train, k) per view
```

Before solving, every view's operator is truncated to its own `n_operator_components`
smallest-eigenvalue directions (default `max(4 * latent_dimensions, 10)`) -- the same truncation
every spectral method already applies, and effectively this class's regularisation strength.
Set too large (approaching `n_samples - 1`), the joint eigenproblem hands each view as many free
directions as training points and, like any unregularised multivariate CCA at that
dimensionality-to-sample-size ratio, starts fabricating cross-view correlation out of pure noise;
set too small, it can discard real manifold structure. On two identical views, with this in
place, `ManifoldCCA` reduces *exactly* to plain single-view spectral embedding of that view --
the concrete check that this is the natural multiview generalisation of `SpectralEmbedding`,
not an unrelated construction that happens to reuse its graph.

### Transform

There is no feature map to apply to new data, so out-of-sample projection reuses each method's
own established extension rather than a generic auxiliary model:

- `method="lle"`: a new point's barycentric reconstruction weights against its `n_neighbors`
  nearest *training* points, applied to those points' rows of the fitted embedding -- exactly
  `LocallyLinearEmbedding.transform`'s own mechanism (verified directly against it in the tests).
- `method="laplacian"`: the classical Nystrom extension (Bengio et al. 2003) of each kept
  eigenvector, using the same degree-normalised affinity rule the training graph was built from,
  then the same combination the joint solve used at training time.

```python
z1, z2 = model.transform([X1_test, X2_test])
```

`inverse_transform`/`predict` are not supported (same limitation as `KCCA`): both assume a
`(n_features_i, k)` weight matrix, not a `(n_train_samples, k)` transductive embedding.

**Not implemented:** Hessian-LLE and LTSA (`sklearn.manifold.LocallyLinearEmbedding`'s other two
`method` options) need a local Hessian/tangent-space estimate per point, substantially more
involved to get right than the Laplacian or plain LLE operator. Isomap-flavoured (geodesic
distance) regularisation is achievable today via `KCCA` with a precomputed geodesic Gram matrix
in place of a standard kernel.

---

## Hyperparameter tuning

Kernel hyperparameters (`c`, `gamma`, `degree`) are best selected by cross-validation.
Use `GridSearchCV` from `cca_zoo.model_selection`:

```python
from cca_zoo.model_selection import GridSearchCV
from cca_zoo.nonparametric import KCCA

param_grid = {
    "c": [0.01, 0.1, 1.0],
    "gamma": [0.001, 0.01, 0.1],
}
gs = GridSearchCV(
    KCCA(latent_dimensions=2, kernel="rbf"),
    param_grid=param_grid,
    cv=5,
)
gs.fit([X1, X2])
print("Best params:", gs.best_params_)
best_model = gs.best_estimator_
```

---

## Custom kernels

Pass any callable with signature `k(X, Y, **params) -> np.ndarray`:

```python
import numpy as np
from cca_zoo.nonparametric import KCCA


def my_kernel(X, Y, sigma=1.0):
    """Gaussian kernel with explicit sigma."""
    diff = X[:, None, :] - Y[None, :, :]
    return np.exp(-np.sum(diff**2, axis=-1) / (2 * sigma**2))


model = KCCA(
    latent_dimensions=2,
    kernel=my_kernel,
    kernel_params={"sigma": 0.5},
    c=0.1,
).fit([X1, X2])
```

---

## Practical notes

- Kernel methods store the full $n \times n$ kernel matrices. Memory is $O(n^2)$; be cautious
  with $n > 10{,}000$.
- `ManifoldCCA` solves a dense $(nM) \times (nM)$ generalised eigenproblem ($n$ = training
  samples, $M$ = number of views) -- the same cost profile as the kernel methods above, intended
  for moderate training-set sizes rather than very large $n$.
- For large datasets, prefer the linear EY-loss methods (`CCAEY`, `PLSEY`, `StochasticCCAEY`)
  or deep methods.
- The `c` parameter is crucial: too small → numerical instability; too large → loss of structure.
  Use cross-validation (see [Model Selection](model-selection.md)).
