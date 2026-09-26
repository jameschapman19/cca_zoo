# Sparse Methods

The `cca_zoo.sparse` module provides sparse/regularised linear multiview CCA methods, from two
mechanism families. Three are EY-loss coordinate descent, each the EY-loss analogue of a
scikit-learn regularised linear regressor: `ElasticNetCCA` (elastic net),
`MultiTaskElasticNetCCA` (multi-task elastic net — row-group sparsity across latent dimensions),
and `OrthogonalMatchingPursuitCCA` (greedy fixed-cardinality selection). The remaining seven —
`PMDCCA`, `ADMMCCA`, `IPLSCCA`, `WaijenborgCCA`, `ParkhomenkoCCA`, `SpanCCA`, `SAR` — are
Alternating Least Squares (ALS) methods, each a from-the-literature sparse CCA algorithm with its
own penalty and fitting loop; see [below](#alternating-least-squares-methods). None has an
optional dependency: all are built entirely on numpy, already required by `cca_zoo`.

---

## Background

`ElasticNetCCA` minimises the same unconstrained Eckart-Young (EY) objective used by the
`CCAEY`/`PLSEY` models in `cca_zoo.linear` and `StochasticCCAEY` in `cca_zoo.stochastic`, by
`DCCAEY` in `cca_zoo.deep`, by `TreeCCA` in `cca_zoo.tree`, and by
`GAMCCA`/`GaussianProcessCCA` (all share the exact same implementation, in
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
`model.weights_` holds real per-view canonical weight matrices, which `TreeCCA`, `GAMCCA` and
`GaussianProcessCCA` don't have.

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
active_features = (model.weights_[0] != 0).any(axis=1)  # same mask for every component
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
active_features = (model.weights_[0] != 0).any(axis=1)
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
corr = model.score([X1, X2])  # mean canonical correlation

# ElasticNetCCA also supports more than two views
model3 = ElasticNetCCA(latent_dimensions=2, alpha=0.1).fit([X1, X2, X3])
```

`l1_ratio=0` is pure ridge (no sparsity); `l1_ratio=1` is pure lasso. Increasing `alpha` shrinks
more weights to exactly zero:

```python
model = ElasticNetCCA(latent_dimensions=1, alpha=0.5, l1_ratio=0.9).fit([X1, X2])
nonzero_features = (model.weights_[0] != 0).any(axis=1)
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

---

## Alternating Least Squares methods

These seven methods use an **Alternating Least Squares (ALS)** loop with Gram-Schmidt deflation
to extract multiple canonical directions, rather than EY-loss coordinate descent.

!!! tip "Choosing an ALS method"
    - **PMDCCA** — fast, interpretable L1 bound; good default for sparse CCA
    - **ADMMCCA** — more principled L1 penalty via ADMM
    - **IPLSCCA** — elastic net penalty; handles both L1 and L2 regularisation
    - **WaijenborgCCA** — elastic net applied to the multiview sum-of-scores target
    - **ParkhomenkoCCA** — simple fixed soft-threshold; fast but less adaptive
    - **SpanCCA** — hard threshold (top-k entries); useful when sparsity level is known
    - **SAR** — penalty strength chosen automatically by BIC; no sparsity hyperparameter to tune

### PMDCCA

Imposes L1 constraints via bisection-based soft-thresholding (Witten 2009):

$$
\max_{\mathbf{w}_1, \mathbf{w}_2} \; \mathbf{w}_1^\top X_1^\top X_2 \mathbf{w}_2
\quad \text{s.t.} \quad \|\mathbf{w}_i\|_1 \leq \tau_i\sqrt{p_i},\; \|\mathbf{w}_i\|_2 = 1
$$

`tau=1` (default) gives no sparsity; smaller values give sparser solutions.

```python
from cca_zoo.sparse import PMDCCA

model = PMDCCA(latent_dimensions=2, tau=0.5, random_state=0).fit([X1, X2])
```

### ADMMCCA

Maximises the cross-view covariance directly, subject to an L1 penalty on each weight
vector and a unit-ball constraint on each view's *score* (Suo, Mineiro & Anandkumar
2017):

$$
\max_{\mathbf{w}_1, \mathbf{w}_2} \; \mathbf{w}_1^\top X_1^\top X_2 \mathbf{w}_2
    - \tau_1\|\mathbf{w}_1\|_1 - \tau_2\|\mathbf{w}_2\|_1
\quad \text{s.t.} \quad \|X_i\mathbf{w}_i\|_2 \leq 1
$$

solved via a linearised Alternating Direction Method of Multipliers, needed because the
constraint couples $\mathbf{w}_i$ to $X_i\mathbf{w}_i$ through a linear map rather than
the identity.

```python
from cca_zoo.sparse import ADMMCCA

model = ADMMCCA(latent_dimensions=2, tau=0.1, random_state=0).fit([X1, X2])
```

### IPLSCCA

Uses an elastic net regression (sklearn) at each ALS step (Mai & Zhang 2019).
`alpha` controls overall regularisation; `l1_ratio=1` gives Lasso, `l1_ratio=0` gives Ridge.

```python
from cca_zoo.sparse import IPLSCCA

model = IPLSCCA(latent_dimensions=2, alpha=0.01, l1_ratio=1.0, random_state=0).fit(
    [X1, X2]
)
```

### WaijenborgCCA

Elastic net CCA (Waaijenborg 2008). Each weight vector is estimated by regressing
the sum-of-all-other-view scores against the current view via elastic net. Named after
the paper's author to disambiguate it from `ElasticNetCCA` above (a different algorithm: an
elastic-net penalty on the actual Eckart-Young CCA loss, not an alternating-regression
heuristic).

```python
from cca_zoo.sparse import WaijenborgCCA

model = WaijenborgCCA(
    latent_dimensions=2, alpha=0.01, l1_ratio=0.5, random_state=0
).fit([X1, X2])
```

### ParkhomenkoCCA

Fixed soft-threshold applied after each power step (Parkhomenko 2009). Simpler than PMD
but `tau` is a fixed threshold, not an L1 bound.

```python
from cca_zoo.sparse import ParkhomenkoCCA

model = ParkhomenkoCCA(latent_dimensions=2, tau=0.1, random_state=0).fit([X1, X2])
```

### SpanCCA

Hard-thresholding retaining only the top `span` entries, an ALS heuristic
inspired by Asteris et al.'s SpanCCA (2016) rather than a reimplementation
of its own randomized low-rank sampling algorithm (this class shares that
paper's algorithm name, not its method). Useful when the number of active
features is known in advance.

```python
from cca_zoo.sparse import SpanCCA

model = SpanCCA(latent_dimensions=2, span=10, random_state=0).fit([X1, X2])
```

### SAR

Sparse Alternating Regression (Wilms & Croux 2015): the same alternating-regression
structure as WaijenborgCCA, but the lasso penalty at each step is picked automatically
by BIC rather than left as a hyperparameter, so there is no `alpha`/`tau`/`span` to
tune. Latent dimensions beyond the first need an extra re-expression step a lasso fit
requires and an OLS-based one does not (see the class docstring for why).

```python
from cca_zoo.sparse import SAR

model = SAR(latent_dimensions=2, random_state=0).fit([X1, X2])
```
