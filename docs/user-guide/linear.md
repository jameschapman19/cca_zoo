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

## EY-loss methods

These methods replace the full eigendecomposition with the unconstrained Eckart-Young (EY)
objective (see [`cca_zoo.tree`](tree.md) and [`cca_zoo.deep`](deep.md) for the same objective
applied to tree and neural-network encoders), making them practical for very high-dimensional
data. No manifold projection step and no upfront whitening pass over the full dataset are
needed: the EY loss's quadratic penalty term drives the weights towards the canonical
directions on its own.

| Class | Description |
|---|---|
| `PLSEY` | Eckart-Young PLS objective, full-batch L-BFGS-B |
| `CCAEY` | Eckart-Young CCA for 2 or more views, full-batch L-BFGS-B, ridge-blended with `PLSEY` via `c` |
| `HuberCCA` | Bounded-influence (Huber-style) EY-CCA, full-batch L-BFGS-B |

For datasets too large to fit comfortably in memory, see
[`cca_zoo.stochastic.StochasticCCAEY`](stochastic.md), which fits the same
objective as `CCAEY` with mini-batch momentum SGD instead.

`CCAEY`'s `c` parameter (default `0`) blends its loss towards `PLSEY`'s (`c=1`) — in fact
`PLSEY` is implemented as `CCAEY` with `c` fixed at `1`. Optimising the raw, unregularised
(`c=0`) objective can be poorly conditioned when the number of samples doesn't outnumber the
number of features by a healthy margin; if you see `nan` weights, increase `c` (0.1-0.3 is
usually enough).

```python
from cca_zoo.linear import CCAEY

model = CCAEY(latent_dimensions=2, max_iter=200)
model.fit([X1, X2])
```

### HuberCCA — bounded-influence EY-CCA

`CCAEY`'s cross- and auto-covariance statistics weight every sample equally, so a handful of
high-leverage points (their contribution grows with the *square* of their magnitude) can hijack
the fit. `HuberCCA` reweights each sample by a Huber-style factor of its own leverage before
forming those statistics — the same bounded-influence mechanism
`sklearn.linear_model.HuberRegressor` uses against outliers, applied to the EY loss's own
statistics. Every sample still contributes *something* (smooth downweighting, never exactly
zero). Fit by the same full-batch L-BFGS-B as `CCAEY`, so it shares that class's `nan`-on
ill-conditioned-data caveat above; it has no ridge-blend `c` of its own. The `delta` parameter
sets the cutoff as a multiple of the dataset's own median sample leverage (self-calibrating, so
it doesn't need re-tuning per `latent_dimensions`); values below 1 downweight the majority of the
data and are not recommended.

```python
from cca_zoo.linear import HuberCCA

model = HuberCCA(latent_dimensions=2, delta=4.0, max_iter=200)
model.fit([X1, X2])
```

---

## Robust methods

`HuberCCA` above guards against high-*leverage* contamination: points whose combined magnitude
across views dominates the covariance statistics simply by being large. That leaves a different
failure mode untouched — a subset of rows whose cross-view *relationship* is wrong (mismatched,
corrupted, or drawn from an unrelated pattern) while remaining completely ordinary in magnitude
within each view. Nothing about such a row's norm flags it as unusual, so leverage-based
reweighting can't see it, and in practice can even make things slightly worse.

`RANSACCCA` is the multiview-CCA analogue of `sklearn.linear_model.RANSACRegressor`: it repeatedly
fits a fast closed-form `MCCA` on a random subset of rows, scores each candidate by how much of
the *full* dataset actually agrees with it, and keeps the best-supported candidate's consensus set
for a final refit. Since it measures agreement with a candidate direction rather than raw
magnitude, it catches exactly the contamination `HuberCCA` can't:

```python
from cca_zoo.linear import RANSACCCA

model = RANSACCCA(latent_dimensions=2, min_samples=0.25, random_state=0)
model.fit([X1, X2])
inliers = model.inlier_mask_  # boolean array over the training rows
```

`min_samples` (a fraction or an absolute count) trades off two things: smaller subsets are more
likely to be drawn free of contamination, but need `c` (a small ridge, default `0.1`) to stay
numerically well-posed. `residual_threshold` defaults to `0` — the natural zero point of the
per-sample agreement score under no real relationship — rather than anything estimated from the
data. Like `MCCA`, this isn't convex, and the random subset draws add their own instability on top:
when the "wrong" relationship is supported by close to half the data, different `random_state`
seeds can land on different consensus sets — see the class's own tests for a worked example of
where this helps and where the problem becomes too ambiguous for any method to resolve reliably.

`RANSACCCA`'s random-subset search is exactly where it struggles too: near the ~50% contamination
breakdown point, a random `min_samples`-sized draw becomes close to a coin flip on being usably
clean, however many trials are tried. `TrimmedCCA` takes a different approach borrowed from
Rousseeuw's Least Trimmed Squares / Minimum Covariance Determinant: rather than gambling on a lucky
small draw, it starts from a large random subset of `h_frac * n` rows and alternates *concentration
steps* — rank every row by its own contribution to `CCAEY`'s exact loss and keep the best `h`, then
re-fit on exactly those rows — each step provably non-increasing in the real loss. With `h_frac` set
close to the true clean fraction, this holds up where `RANSACCCA`'s search degrades:

```python
from cca_zoo.linear import TrimmedCCA

model = TrimmedCCA(h_frac=0.55, n_starts=40, random_state=0)
model.fit([X1, X2])
inliers = model.inlier_mask_  # boolean array over the training rows
```

`h_frac` is a prior on the contamination rate, not something fit from the data — set it too high and
good rows get discarded for nothing; set it too low and contaminated rows get forced into every fit
once true contamination exceeds `1 - h_frac`. `TrimmedCCA` supports any number of views (2 or more)
but only `latent_dimensions=1`: the selection rule's closed-form derivation relies on `CCAEY`'s
penalty being the square of a *single* linear functional of the selection, which holds for any
number of views but not past one latent dimension — with `k > 1` the same penalty becomes a genuine
matrix-valued quadratic form (rank up to `k(k+1)/2`) that the same single-multiplier bisection can't
solve. Away from the ~50% breakdown regime, or when more than one latent dimension is needed,
`RANSACCCA` matches or beats it directly.

---

## Choosing a method

| Scenario | Recommended |
|---|---|
| $n \gg p$, two views | `CCA` |
| $n < p$ or ill-conditioned | `rCCA` (tune `c`) |
| Maximise covariance, not correlation | `PLS` |
| Three or more views | `MCCA` or `GCCA` |
| Higher-order cross-view structure | `TCCA` |
| Sparse weights needed | [`cca_zoo.sparse.PMDCCA`](sparse.md) or [`IPLSCCA`](sparse.md) |
| Very large $p$ | `CCAEY`, `PLSEY` |
| Dataset too large for full-batch gradients | [`cca_zoo.stochastic.StochasticCCAEY`](stochastic.md) |
| A few high-magnitude outlier samples | `HuberCCA` |
| A subset of rows with a wrong (mismatched/corrupted) relationship | `RANSACCCA` |
| Heavy contamination (near ~50%), with a known contamination-rate prior | `TrimmedCCA` |
| Nonlinear relationships | See [Nonparametric Methods](nonparametric.md) |
