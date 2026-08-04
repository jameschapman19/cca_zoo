# Linear Methods

All linear CCA methods in `cca_zoo.linear` share the same fit/transform/score interface and are
`sklearn.base.BaseEstimator` subclasses. They find **linear** projections of the input views.

---

## Two-view methods

These methods operate on exactly two views.

### CCA — Standard Canonical Correlation Analysis

**When to use:** The default choice for two balanced, moderately-sized views.

CCA finds directions $\mathbf{w}_1, \mathbf{w}_2$ that maximise the Pearson correlation between
projected views:

$$
\max_{\mathbf{w}_1, \mathbf{w}_2} \; \mathbf{w}_1^\top X_1^\top X_2 \mathbf{w}_2
\quad \text{s.t.} \quad \mathbf{w}_i^\top X_i^\top X_i \mathbf{w}_i = 1
$$

The solution uses PCA whitening followed by an SVD of the cross-covariance matrix, giving a
numerically stable result even for high-dimensional views.

```python
from cca_zoo.linear import CCA

model = CCA(latent_dimensions=2).fit([X1, X2])
z1, z2 = model.transform([X1, X2])
print(model.score([X1, X2]))  # canonical correlations
```

### rCCA — Regularised CCA

**When to use:** CCA breaks down when $n < p$ (more features than samples). rCCA adds a ridge
penalty to stabilise the covariance matrices.

The parameter `c` controls the regularisation strength:

- `c=0` → equivalent to `CCA`
- `c=1` → equivalent to `PLS`
- `0 < c < 1` → interpolates between the two

```python
from cca_zoo.linear import rCCA

model = rCCA(latent_dimensions=2, c=0.1).fit([X1, X2])
```

### PLS — Partial Least Squares

**When to use:** When you want to maximise *covariance* rather than *correlation*. PLS is
more robust to noise and does not require invertible covariance matrices.

PLS is a special case of rCCA with `c=1`:

$$
\max_{\mathbf{w}_1, \mathbf{w}_2} \; \mathbf{w}_1^\top X_1^\top X_2 \mathbf{w}_2
\quad \text{s.t.} \quad \|\mathbf{w}_i\|_2 = 1
$$

```python
from cca_zoo.linear import PLS

model = PLS(latent_dimensions=2).fit([X1, X2])
```

---

## Multiview methods (≥2 views)

These methods generalise CCA to three or more views.

### MCCA — Multiset CCA

**When to use:** Multiple views, interpretable pairwise-sum objective.

MCCA maximises the sum of pairwise correlations across all view pairs. It solves a generalised
eigenvalue problem on block matrices:

$$
A \mathbf{v} = \lambda B \mathbf{v}
$$

where $A$ contains the cross-view covariances and $B$ the regularised within-view covariances.

```python
from cca_zoo.linear import MCCA

model = MCCA(latent_dimensions=2, c=0.1).fit([X1, X2, X3])
```

### GCCA — Generalised CCA

**When to use:** Multiple views with potentially different numbers of features; best when you
want a single shared low-dimensional representation.

GCCA finds a common latent variable $G$ (of shape $n \times k$) such that each view can be
reconstructed from it:

$$
\min_{G, W_i} \sum_{i} \mu_i \|X_i W_i - G\|_F^2
$$

This is equivalent to maximising the sum of squared canonical correlations between each view and
the shared projection.

```python
from cca_zoo.linear import GCCA

model = GCCA(latent_dimensions=2, c=0.01).fit([X1, X2, X3])
```

### TCCA — Tensor CCA

**When to use:** When you want to capture higher-order (beyond pairwise) correlations among
multiple views.

TCCA builds a joint cross-moment tensor of the whitened views and finds its best rank-$k$
PARAFAC decomposition:

$$
T = \frac{1}{n} \sum_{j} \tilde{\mathbf{x}}^{(1)}_j \otimes \cdots \otimes \tilde{\mathbf{x}}^{(M)}_j
$$

```python
from cca_zoo.linear import TCCA

model = TCCA(latent_dimensions=2, c=0.01, random_state=0).fit([X1, X2, X3])
```

---

## Gradient-descent methods

These methods replace the full eigendecomposition with mini-batch momentum gradient descent on
the unconstrained Eckart-Young (EY) objective (see [`cca_zoo.tree`](tree.md) and
[`cca_zoo.deep`](deep.md) for the same objective applied to tree and neural-network encoders),
making them practical for very high-dimensional or streaming data. No manifold projection step
and no upfront whitening pass over the full dataset are needed: the EY loss's quadratic penalty
term drives the weights towards the canonical directions on its own, directly from raw
mini-batches.

| Class | Description |
|---|---|
| `PLS_EY` | Eckart-Young PLS objective, stochastic updates |
| `CCA_EY` | Eckart-Young CCA, stochastic updates, ridge-blended with `PLS_EY` via `c` |
| `MCCA_EY` | Multiview EY-CCA for ≥2 views |

`CCA_EY`'s `c` parameter (default `0`) blends its loss towards `PLS_EY`'s (`c=1`) — in fact
`PLS_EY` is implemented as `CCA_EY` with `c` fixed at `1`. Gradient descent on the raw,
unregularised (`c=0`) objective can diverge when a mini-batch's samples don't outnumber the
number of features by a healthy margin; if you see `nan` weights, increase `c` (0.1-0.3 is
usually enough) or `batch_size`.

```python
from cca_zoo.linear import CCA_EY

model = CCA_EY(latent_dimensions=2, learning_rate=0.01, batch_size=128, max_iter=200)
model.fit([X1, X2])
```

---

## Sparse / iterative methods

All sparse methods in `cca_zoo.linear` use an **Alternating Least Squares (ALS)** loop with
Gram-Schmidt deflation to extract multiple canonical directions.

!!! tip "Choosing a sparse method"
    - **SCCA_PMD** — fast, interpretable L1 bound; good default for sparse CCA
    - **SCCA_ADMM** — more principled L1 penalty via ADMM
    - **SCCA_IPLS** — elastic net penalty; handles both L1 and L2 regularisation
    - **ElasticCCA** — elastic net applied to the multiview sum-of-scores target
    - **ParkhomenkoCCA** — simple fixed soft-threshold; fast but less adaptive
    - **SCCA_Span** — hard threshold (top-k entries); useful when sparsity level is known
    - **PLS_ALS** — no sparsity; ALS version of PLS (useful as a baseline)

### SCCA_PMD

Imposes L1 constraints via bisection-based soft-thresholding (Witten 2009):

$$
\max_{\mathbf{w}_1, \mathbf{w}_2} \; \mathbf{w}_1^\top X_1^\top X_2 \mathbf{w}_2
\quad \text{s.t.} \quad \|\mathbf{w}_i\|_1 \leq \tau_i\sqrt{p_i},\; \|\mathbf{w}_i\|_2 = 1
$$

`tau=1` (default) gives no sparsity; smaller values give sparser solutions.

```python
from cca_zoo.linear import SCCA_PMD

model = SCCA_PMD(latent_dimensions=2, tau=0.5, random_state=0).fit([X1, X2])
```

### SCCA_ADMM

Solves the same L1-constrained problem via the Alternating Direction Method of Multipliers
(Suo 2017). Often more precise than PMD for tight sparsity budgets.

```python
from cca_zoo.linear import SCCA_ADMM

model = SCCA_ADMM(latent_dimensions=2, tau=0.1, random_state=0).fit([X1, X2])
```

### SCCA_IPLS

Uses an elastic net regression (sklearn) at each ALS step (Mai & Zhang 2019).
`alpha` controls overall regularisation; `l1_ratio=1` gives Lasso, `l1_ratio=0` gives Ridge.

```python
from cca_zoo.linear import SCCA_IPLS

model = SCCA_IPLS(latent_dimensions=2, alpha=0.01, l1_ratio=1.0, random_state=0).fit(
    [X1, X2]
)
```

### ElasticCCA

Elastic net CCA (Waaijenborg 2008). Each weight vector is estimated by regressing
the sum-of-all-other-view scores against the current view via elastic net.

```python
from cca_zoo.linear import ElasticCCA

model = ElasticCCA(latent_dimensions=2, alpha=0.01, l1_ratio=0.5, random_state=0).fit(
    [X1, X2]
)
```

### ParkhomenkoCCA

Fixed soft-threshold applied after each power step (Parkhomenko 2009). Simpler than PMD
but `tau` is a fixed threshold, not an L1 bound.

```python
from cca_zoo.linear import ParkhomenkoCCA

model = ParkhomenkoCCA(latent_dimensions=2, tau=0.1, random_state=0).fit([X1, X2])
```

### SCCA_Span

Hard-thresholding retaining only the top `span` entries (Asteris 2016). Useful when the
number of active features is known in advance.

```python
from cca_zoo.linear import SCCA_Span

model = SCCA_Span(latent_dimensions=2, span=10, random_state=0).fit([X1, X2])
```

### PLS_ALS

Standard ALS/power-iteration variant of PLS without regularisation. Useful as a baseline or
when data are already low-dimensional.

```python
from cca_zoo.linear import PLS_ALS

model = PLS_ALS(latent_dimensions=2, random_state=0).fit([X1, X2])
```

---

## Choosing a method

| Scenario | Recommended |
|---|---|
| $n \gg p$, two views | `CCA` |
| $n < p$ or ill-conditioned | `rCCA` (tune `c`) |
| Maximise covariance, not correlation | `PLS` |
| Three or more views | `MCCA` or `GCCA` |
| Higher-order cross-view structure | `TCCA` |
| Sparse weights needed | `SCCA_PMD` or `SCCA_IPLS` |
| Very large $p$ / streaming data | `CCA_EY`, `PLS_EY` |
| Nonlinear relationships | See [Nonparametric Methods](nonparametric.md) |
