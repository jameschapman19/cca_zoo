# GP Methods

The `cca_zoo.gp` module provides `GPCCA`, a nonlinear multiview CCA method that uses a
Gaussian process with a joint (non-additive) kernel as the per-view encoder. It has no optional
dependency: the kernel and its fit are built entirely from scikit-learn's own
`GaussianProcessRegressor`, `RBF` and `ConstantKernel`, all already required by `cca_zoo`.

---

## Background

`GPCCA` maximises the same unconstrained Eckart-Young (EY) objective used by the stochastic
`*_EY` models in `cca_zoo.linear`, by `DCCAEY` in `cca_zoo.deep`, by `TreeCCA` in `cca_zoo.tree`,
and by `GAMCCA` in `cca_zoo.gam` (all share the exact same implementation, in
`cca_zoo._utils._ey`):

$$
\mathcal{L}_{EY} = -2 \operatorname{tr}(C) + \operatorname{tr}(V V)
$$

where, for embeddings $Z_i = f_i(X_i)$, $C$ is the mean pairwise cross-covariance (including
$i = j$ terms) and $V$ the mean auto-covariance across all views. Where `GAMCCA` sums one
univariate B-spline term per input feature — additive, and so structurally unable to represent an
interaction between two features of the same view — `GPCCA` fits a Gaussian process with an ARD
(per-feature-lengthscale) RBF kernel directly over each view's *raw, joint* feature vector: a
genuine, non-additive function of all of that view's features at once. As a Bayesian model it
also comes with calibrated predictive uncertainty for free.

`GPCCA` is fit with the same inner/outer split as `GAMCCA`'s P-IRLS/GCV recipe, with the GP's own
machinery in place of `Ridge`/`RidgeCV`:

1. **Inner loop — fixed-kernel Newton steps.** For the *current* kernel hyperparameters,
   repeatedly take a Newton step on the EY loss for each view in turn: form a working response
   $Z_i - \nabla_i / h_i$ from the analytic EY gradient $\nabla_i$ and a diagonal Hessian weight
   $h_i$, and fit it with `GaussianProcessRegressor` (`optimizer=None`, so the kernel is held
   fixed), passing $h_i$ as the GP's per-sample `alpha` (heteroscedastic observation noise). Cycle
   through every view until the EY loss itself stops moving.
2. **Outer loop — marginal-likelihood kernel search.** Only once the inner loop has converged,
   re-fit each view's kernel hyperparameters (lengthscales, signal variance) at that converged
   working response via the GP's own default log-marginal-likelihood optimisation — the direct GP
   analogue of GCV/REML smoothing-parameter search in `GAMCCA`. Repeat until the kernel
   hyperparameters stabilise too.

**When to use:** Nonlinear multiview CCA where cross-view structure depends on a genuine
*interaction* between two or more features of the same view (e.g. $x_1 x_2$), which an additive
GAM cannot represent and which a tree ensemble can only approximate via multivariate splits (and,
at default settings, may be starved of joint feature access — see `TreeCCA`'s `colsample_bytree`).
A GP with a joint kernel represents such an interaction directly. When the true relationship is
smooth and *additive*, `GAMCCA` is typically a cheaper and equally accurate choice; a GP over the
raw feature vector scales cubically in the number of training samples (exact GP inference), so
expect fitting to be markedly slower than `GAMCCA` or `TreeCCA` on large datasets.

---

## Basic usage

```python
from cca_zoo.gp import GPCCA

model = GPCCA(latent_dimensions=1).fit([X1, X2])
z1, z2 = model.transform([X1, X2])
corrs = model.score([X1, X2])

# GPCCA also supports more than two views
model3 = GPCCA(latent_dimensions=1).fit([X1, X2, X3])
```

## Predictive uncertainty

Because each per-component encoder is a Gaussian process, `transform` can return each latent
component's posterior standard deviation alongside its mean:

```python
means, stds = model.transform([X1, X2], return_std=True)
```

`stds[i]` has the same shape as `means[i]` (`(n_samples, latent_dimensions)`) and is the posterior
standard deviation of view `i`'s latent component, propagated through the whitening transform —
larger away from the training data, smaller near it, exactly as for any other GP posterior.

`GPCCA` has no linear weight matrices and no per-feature decomposition analogous to `GAMCCA`'s
`shape_function` (the kernel is not additive across features), so `model.weights` raises
`NotImplementedError`.

---

## Scaling to larger datasets: the sparse (inducing-point) approximation

Exact GP inference costs $O(n^3)$ per Newton step — every `inner_step`/`outer_step` call redoes a
full Cholesky factorisation of an $(n, n)$ matrix — which becomes impractical somewhere in the
low thousands of samples. Setting `n_inducing` switches each encoder to the **Deterministic
Training Conditional (DTC)** sparse approximation (Quiñonero-Candela & Rasmussen, 2005): the GP is
conditioned on `n_inducing` inducing points — an actual, well-spread subset of the training rows,
selected via `sklearn.cluster.kmeans_plusplus`'s seeding — and every posterior formula becomes a
function of the $(n, m)$ and $(m, m)$ inducing-covariance matrices instead of the full $(n, n)$
one, costing $O(n \, m^2 + m^3)$ instead of $O(n^3)$:

```python
model = GPCCA(latent_dimensions=1, n_inducing=200).fit([X1, X2])  # X1, X2 have many samples
```

Kernel hyperparameters are still selected by an *exact* marginal-likelihood fit — just on the
`n_inducing` inducing rows alone, where it's cheap — while the actual working-response fit used to
build each round's representation always uses *every* training row via the DTC formula, so no
training signal is thrown away at the point that matters most. `n_inducing` values at or above the
number of training samples are equivalent to (and internally fall back to) exact inference. Larger
`n_inducing` trades speed for a closer approximation to the exact posterior; there's no universal
default, since how many inducing points are "enough" depends on how smooth/low-rank the true
underlying function is — start with a few hundred and check whether increasing it changes the
held-out canonical correlation.

---

## Key parameters

| Parameter | Description |
|---|---|
| `max_inner_iter` | Cap on fixed-kernel Newton rounds (one cycle through every view) per outer iteration. In practice the inner loop converges well before this in typical cases. |
| `max_outer_iter` | Cap on kernel-hyperparameter re-selection rounds. |
| `tol` | Inner-loop convergence tolerance, on the change in the EY loss between successive full passes over all views. |
| `hess_floor_percentile` | Percentile (0-100) of each round's raw diagonal-Hessian values used to floor them — self-calibrating damping against the Hessian's poor conditioning near the loss's own fixed point (see `GAMCCA`'s class docstring for why). Default is 90. |
| `n_inducing` | Number of inducing points for the sparse DTC approximation (see above). `None` (default) uses exact GP inference. |
| `random_state` | Seed for the random-orthogonal initial embedding, and (if `n_inducing` is set) for selecting inducing points. |

---

## Practical notes

- `GPCCA` supports 2 or more views.
- `latent_dimensions` must not exceed the number of features in any view (the random-orthogonal
  initialisation draws that many orthogonal directions in feature space).
- Exact GP inference (`n_inducing=None`) is $O(n^3)$ in the number of training samples; set
  `n_inducing` for datasets beyond a few thousand samples, or consider `GAMCCA` (if the
  relationship is additive) or `TreeCCA` instead.
- No optional dependency is required (unlike `cca_zoo.tree`, which needs `xgboost`/`lightgbm`):
  `GPCCA` is built entirely on `scikit-learn`'s `GaussianProcessRegressor`, `RBF`, `ConstantKernel`
  and `kmeans_plusplus`.
