# GP Methods

The `cca_zoo.gp` module provides `GaussianProcessCCA`, a nonlinear multiview CCA method that uses a
Gaussian process with a joint (non-additive) kernel as the per-view encoder. It has no optional
dependency: the kernel machinery is built entirely from scikit-learn's own
`GaussianProcessRegressor`, `RBF`, `ConstantKernel` and `KernelCenterer`, with `scipy.optimize`
doing the fit, all already required by `cca_zoo`.

---

## Background

`GaussianProcessCCA` minimises the same unconstrained Eckart-Young (EY) objective used by the stochastic
`*_EY` models in `cca_zoo.linear`, by `DCCAEY` in `cca_zoo.deep`, by `TreeCCA` in `cca_zoo.tree`,
and by `GAMCCA` in `cca_zoo.gam` (all share the exact same implementation, in
`cca_zoo._utils._ey`):

$$
\mathcal{L}_{EY} = -2 \operatorname{tr}(C) + \operatorname{tr}(V V)
$$

where, for embeddings $Z_i = f_i(X_i)$, $C$ is the mean pairwise cross-covariance (including
$i = j$ terms) and $V$ the mean auto-covariance across all views. Where `GAMCCA` sums one
univariate B-spline term per input feature — additive, and so structurally unable to represent an
interaction between two features of the same view — `GaussianProcessCCA` fits a Gaussian process with an ARD
(per-feature-lengthscale) RBF kernel directly over each view's *raw, joint* feature vector: a
genuine, non-additive function of all of that view's features at once. As a Bayesian model it
also comes with calibrated predictive uncertainty for free.

Each encoder writes $f_i(x) = k(x, Z_i)^\top B_i$ for a fixed kernel $k$ and a fixed set of basis
("inducing") points $Z_i$ — every training row by default, or `n_inducing` of them selected via
`sklearn.cluster.kmeans_plusplus` for larger datasets. Fitting the coefficients $B_i$ is then an
ordinary smooth optimisation problem with an exact, cheap analytic gradient, solved directly by
**L-BFGS-B** (`scipy.optimize.minimize`) — the same algorithm `GaussianProcessRegressor` itself
uses internally, just pointed at the EY loss (plus an RKHS-norm ridge penalty) instead of the
negative log-marginal-likelihood it optimises kernel hyperparameters against. Every view's
coefficients are optimised **jointly**, in a single L-BFGS-B run over all of them concatenated,
rather than one view at a time with the others held fixed: the EY loss already couples every view
together, so solving one view to convergence before moving to the next needlessly repeats work and
can settle into a worse joint optimum than optimising every view simultaneously against the exact
joint gradient. Kernel hyperparameters are fixed — pass `kernel=` explicitly, or tune it
externally (e.g. with `sklearn.model_selection.GridSearchCV`, since this is an ordinary
`BaseEstimator`).

**When to use:** Nonlinear multiview CCA where cross-view structure depends on a genuine
*interaction* between two or more features of the same view (e.g. $x_1 x_2$), which an additive
GAM cannot represent and which a tree ensemble can only approximate via multivariate splits (and,
at default settings, may be starved of joint feature access — see `TreeCCA`'s `colsample_bytree`).
A GP with a joint kernel represents such an interaction directly. When the true relationship is
smooth and *additive*, `GAMCCA` is typically a cheaper and equally accurate choice; a GP over the
raw feature vector scales cubically in the number of training samples (exact inference), so
expect fitting to be markedly slower than `GAMCCA` or `TreeCCA` on large datasets unless
`n_inducing` is set.

---

## Basic usage

```python
from cca_zoo.gp import GaussianProcessCCA

model = GaussianProcessCCA(latent_dimensions=1).fit([X1, X2])
z1, z2 = model.transform([X1, X2])
corr = model.score([X1, X2])  # mean canonical correlation

# GaussianProcessCCA also supports more than two views
model3 = GaussianProcessCCA(latent_dimensions=1).fit([X1, X2, X3])
```

## Predictive uncertainty

Because each per-component encoder is a Gaussian process, `transform` can return each latent
component's posterior standard deviation alongside its mean:

```python
means, stds = model.transform([X1, X2], return_std=True)
```

`stds[i]` has the same shape as `means[i]` (`(n_samples, latent_dimensions)`) and is the posterior
standard deviation of view `i`'s latent component — larger away from the training data, smaller
near it, exactly as for any other GP posterior. This does not depend on the fitted coefficients at
all (a standard GP fact: posterior variance only involves the kernel, the noise level, and the
design points), so it is computed under the (uncentred) GP prior implied by the same kernel, noise
level, and inducing points as the mean fit.

`GaussianProcessCCA` has no linear weight matrices and no per-feature decomposition analogous to
`GAMCCA`'s `shape_function` (the kernel is not additive across features). `feature_importances_`
is therefore permutation-based: the mean squared change in each view's latent scores when a
feature's training values are shuffled, normalised to sum to 1 per view.

---

## Scaling to larger datasets: the sparse (inducing-point) approximation

Exact inference costs $O(n^3)$ per fit, using every training row as a basis point — impractical
somewhere in the low thousands of samples. Setting `n_inducing` switches each encoder to a
**reduced-rank ("subset of regressors")** construction (Quiñonero-Candela & Rasmussen, 2005),
using only `n_inducing` basis points — an actual, well-spread subset of the training rows,
selected via `sklearn.cluster.kmeans_plusplus`'s seeding — reducing fitting to
$O(n \, m^2 + m^3)$ for $m$ basis points instead of $O(n^3)$:

```python
model = GaussianProcessCCA(latent_dimensions=1, n_inducing=200).fit(
    [X1, X2]
)  # X1, X2 have many samples
```

`n_inducing` values at or above the number of training samples are equivalent to (and internally
fall back to) exact inference. Larger `n_inducing` trades speed for a closer approximation to the
exact posterior; there's no universal default, since how many inducing points are "enough" depends
on how smooth/low-rank the true underlying function is — start with a few hundred and check
whether increasing it changes the held-out canonical correlation.

---

## Key parameters

| Parameter | Description |
|---|---|
| `kernel` | Fixed kernel used for every view. `None` (default) uses `ConstantKernel(1.0) * RBF(length_scale=np.ones(p))` for each view's own feature count `p`. |
| `alpha` | Ridge (RKHS-norm) penalty strength, also used as the noise level for the posterior-variance calculation. |
| `n_inducing` | Number of basis ("inducing") points (see above). `None` (default) uses every training row (exact inference). |
| `max_iter` | Maximum number of L-BFGS-B iterations for the single, joint solve over every view's coefficients. |
| `tol` | Convergence tolerance, passed to L-BFGS-B as `ftol`. |
| `random_state` | Seed for the initial coefficients, and (if `n_inducing` is set) for selecting inducing points. |

---

## Practical notes

- `GaussianProcessCCA` supports 2 or more views.
- `latent_dimensions` must not exceed the number of features in any view.
- Exact inference (`n_inducing=None`) is $O(n^3)$ in the number of training samples; set
  `n_inducing` for datasets beyond a few thousand samples, or consider `GAMCCA` (if the
  relationship is additive) or `TreeCCA` instead.
- No optional dependency is required (unlike `cca_zoo.tree`, which needs `xgboost`/`lightgbm`):
  `GaussianProcessCCA` is built entirely on `scikit-learn`'s `GaussianProcessRegressor`, `RBF`, `ConstantKernel`,
  `KernelCenterer` and `kmeans_plusplus`.
