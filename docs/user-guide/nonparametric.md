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

Each view's kernel $K_i$ is centred in feature space (sklearn's `KernelCenterer`), and a
new row's kernel against the training rows is centred with the training statistics. KCCA
is then exactly `MCCA` on each view's kernel feature map: with dual variables
$\boldsymbol{\alpha}_i$ it solves

$$
A \boldsymbol{\alpha} = \lambda B \boldsymbol{\alpha}
$$

where:

- $A$ has off-diagonal blocks $K_i K_j / (n-1)$, the covariances between views' scores
- $B$ has diagonal blocks $(1-c_i) K_i^2 / (n-1) + c_i K_i$: the variance of the scores
  $K_i \boldsymbol{\alpha}_i$, shrunk towards the squared norm of the feature-space direction

`shrinkage` ($c_i$) means what it does for `MCCA`: 0 is kernel CCA and 1 kernel PLS. With
`kernel="linear"`, `KCCA`, `KGCCA` and `KTCCA` are `MCCA`, `GCCA` and `TCCA`, and with any
kernel `KCCA` equals `MCCA` on `Nystroem` features that use every training row as a
landmark. For large $n$, a `Nystroem` approximation with fewer landmarks followed by the
linear model is the scalable form of the same estimator.

```python
from cca_zoo.nonparametric import KCCA

# Linear kernel (recovers classical CCA in feature space)
model = KCCA(n_components=2, kernel="linear", shrinkage=0.1).fit([X1, X2])

# RBF kernel
model = KCCA(n_components=2, kernel="rbf", gamma=0.01, shrinkage=0.1).fit([X1, X2])

# Polynomial kernel
model = KCCA(n_components=2, kernel="poly", degree=3, shrinkage=0.1).fit([X1, X2])

# Per-view kernel parameters (list = one entry per view)
model = KCCA(
    n_components=2,
    kernel=["rbf", "poly"],
    gamma=[0.01, None],
    degree=[1, 3],
    shrinkage=[0.1, 0.5],
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

with centred kernels and $B_i = (1-c_i) K_i^2 / (n-1) + c_i K_i$; it is `GCCA` on each view's kernel feature map.

```python
from cca_zoo.nonparametric import KGCCA

model = KGCCA(n_components=2, kernel="rbf", gamma=0.01, shrinkage=0.1).fit([X1, X2, X3])
```

---

## KTCCA — Kernel Tensor CCA

**When to use:** Captures higher-order correlations in the kernel space for three or more views.

KTCCA is `TCCA` on each view's kernel feature map: it whitens the feature maps, builds their
cross-moment tensor and applies PARAFAC decomposition:

```python
from cca_zoo.nonparametric import KTCCA

model = KTCCA(
    n_components=2, kernel="rbf", gamma=0.01, shrinkage=0.1, random_state=0
).fit([X1, X2, X3])
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

model = ManifoldCCA(method="laplacian", n_neighbors=10, n_components=1).fit([X1, X2])
train_embedding = model.embedding_  # (n_train, k) per view
```

Before solving, every view's operator is truncated to its own `n_operator_components`
smallest-eigenvalue directions (default `max(4 * n_components, 40)`) -- the same truncation
every spectral method already applies. Set too small, it discards real manifold structure; set
very large with `shrinkage` near 0, the joint eigenproblem hands each view as many free
directions as training points and, like any unregularised CCA at that dimensionality-to-sample
ratio, can fabricate cross-view correlation. On two identical views with `shrinkage > 0`,
`ManifoldCCA` reduces exactly to that view's LLE embedding, and up to centring to its spectral
embedding -- the concrete check that this is the natural multiview generalisation of
`LocallyLinearEmbedding` and `SpectralEmbedding`, not an unrelated construction that happens to
reuse their graphs.

`shrinkage` works as for `MCCA` and `KCCA`. Each view's constraint blends its embedding's
variance with its roughness on the view's graph, $(1-c)\,\mathrm{Cov}(Z_i, Z_i) + c\,\kappa_i Z_i^\top M_i Z_i$,
where $\kappa_i$ rescales the roughness of the kept eigenvectors to the same trace as their
variance, so that a given `shrinkage` means the same for the LLE operator (whose smooth
eigenvalues are tiny) and the normalised Laplacian (whose are not):
`shrinkage=0` is CCA between the views' smooth graph coordinates, and `shrinkage=1` is PLS,
which maximises covariance per unit of roughness and so trades correlation for smoothness. The
fit is exactly `MCCA(shrinkage=c)` on each view's kept operator eigenvectors, each scaled by the
inverse square root of its rescaled eigenvalue. The default, 0.1, is `KCCA`'s.

```python
model = ManifoldCCA(method="lle", shrinkage=0.01, n_neighbors=10).fit([X1, X2])
```

### Transform

There is no feature map to apply to new data, so out-of-sample projection reuses each method's
own established extension rather than a generic auxiliary model:

- `method="lle"`: a new point's barycentric reconstruction weights against its `n_neighbors`
  nearest *training* points, applied to those points' rows of the fitted embedding -- exactly
  `LocallyLinearEmbedding.transform`'s own mechanism (verified directly against it in the tests).
- `method="laplacian"`: the classical Nystrom extension (Bengio et al. 2003) of each kept
  eigenvector, using the same degree-normalised affinity rule the training graph was built from,
  then the same combination the joint solve used at training time. The extension divides by
  `1 - eigenvalue`, so eigenvectors with eigenvalue above 0.95, barely smoother than noise, are
  left out of the basis rather than amplified.

Either extension returns a training point's own embedding: `transform` of the training views is
`embedding_`.

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

Kernel hyperparameters (`shrinkage`, `gamma`, `degree`) are best selected by cross-validation.
Use `GridSearchCV` from `cca_zoo.model_selection`:

```python
from cca_zoo.model_selection import GridSearchCV
from cca_zoo.nonparametric import KCCA

param_grid = {
    "shrinkage": [0.01, 0.1, 1.0],
    "gamma": [0.001, 0.01, 0.1],
}
gs = GridSearchCV(
    KCCA(n_components=2, kernel="rbf"),
    param_grid=param_grid,
    cv=5,
)
gs.fit([X1, X2])
print("Best params:", gs.best_params_)
best_model = gs.best_estimator_
```

---

## Custom kernels

Pass any callable with signature `k(x, y, **params) -> float`. As with sklearn's
`pairwise_kernels`, it is called on one pair of samples at a time, and `kernel_params`
supplies its keyword arguments:

```python
import numpy as np
from cca_zoo.nonparametric import KCCA


def my_kernel(x, y, sigma=1.0):
    """Gaussian kernel of two samples, with explicit sigma."""
    return np.exp(-np.sum((x - y) ** 2) / (2 * sigma**2))


model = KCCA(
    n_components=2,
    kernel=my_kernel,
    kernel_params={"sigma": 0.5},
    shrinkage=0.1,
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
- The `shrinkage` parameter is crucial: too small → numerical instability; too large → loss of structure.
  Use cross-validation (see [Model Selection](model-selection.md)).
