# Probabilistic CCA

The `cca_zoo.probabilistic` module provides a Bayesian treatment of CCA, with three inference
backends:

- `GFA` — closed-form coordinate-ascent variational Bayes with **per-view** automatic relevance
  determination (ARD). No extra dependencies; always available.
- `ProbabilisticCCA` — full MCMC sampling via NumPyro. Requires the `[probabilistic]` extra.
- `VariationalBayesCCA` — black-box variational inference with ARD **shared** across views, via
  NumPyro. Requires the `[probabilistic]` extra.

```bash
pip install cca-zoo[probabilistic]  # only needed for ProbabilisticCCA / VariationalBayesCCA
```

---

## Background

Classical CCA finds a single point estimate of the canonical weights. **Probabilistic CCA**
(Bach & Jordan 2005) instead defines a generative model with explicit priors, enabling
uncertainty quantification over the weights. `ProbabilisticCCA` and `VariationalBayesCCA` share
this base generative model:

$$
\mathbf{z} \sim \mathcal{N}(\mathbf{0}, I_k)
$$

$$
\mathbf{x}_i \mid \mathbf{z} \sim \mathcal{N}(W_i \mathbf{z},\; \mathrm{diag}(\boldsymbol{\psi}_i))
$$

where:

- $\mathbf{z}$ is the $k$-dimensional shared latent variable
- $W_i$ is the view-specific loading matrix with a Normal prior
- $\boldsymbol{\psi}_i$ are per-feature noise variances with a log-Normal prior

They share this model and the same posterior-mean projection formula for `transform`; they
differ only in how the posterior is approximated. `GFA` modifies the model itself (per-view ARD,
per-view scalar noise instead of per-feature) — see its own section below.

### Scoring: correlation vs likelihood

All three classes implement `score` (average pairwise correlation) for consistency with every
other model in `cca_zoo` — this is what `GridSearchCV` optimizes by default. But a correlation
between projections isn't the statistically natural fit criterion for a *probabilistic* model.
All three also implement `log_likelihood`, the marginal log-likelihood of the data with the
shared latent variable integrated out:

$$
\mathbf{x} \sim \mathcal{N}\!\left(0,\ \Psi + WW^\top\right)
$$

where $\mathbf{x}$ is the concatenation of every view's centred features for one sample, $W$
stacks every view's loading matrix, and $\Psi$ is the (block-)diagonal noise-variance matrix.
This is evaluated jointly across the concatenated views rather than per view: because every view
shares the same $z$, marginalising it induces cross-view covariance that a per-view likelihood
would silently ignore. Use `log_likelihood` to compare `latent_dimensions` choices, or to compare
any of the three classes against each other on the same data — larger (less negative) is better.

```python
model.log_likelihood([X1, X2])  # mean log-likelihood per sample
```

---

## `GFA`: per-view ARD, no extra dependencies

`GFA` (Group Factor Analysis; Klami, Virtanen & Kaski 2013) is ported directly from the
reference R package [`CCAGFA`](https://github.com/cran/CCAGFA) — the update equations are
transliterated from that source, not re-derived. It fits a single shared latent variable $z$,
but gives **each view its own ARD precision** $\alpha_{i,k}$ per latent dimension, rather than
tying every view to the same precision like `VariationalBayesCCA`:

$$
\alpha_{i,k} \sim \mathrm{Gamma}(a_0, b_0), \qquad W_i[:, k] \sim \mathcal{N}(0,\ \alpha_{i,k}^{-1} I)
$$

"Shared" vs. "private" dimensions are therefore *emergent*, not a fixed split of $z$ into
blocks: a dimension ends up shared if its $\alpha_{i,k}$ stays small in several views at once,
and private to view $i$ if it shrinks toward zero loadings in every *other* view. This is the
actual mechanism behind "Bayesian CCA" that distinguishes it from `VariationalBayesCCA`'s single
shared ARD parameter per dimension. It also uses a different noise model: $\tau_i$ is a single
scalar precision per view (homoscedastic), not a per-feature diagonal.

Inference is closed-form coordinate-ascent variational Bayes — fully conjugate, so there's no
need for NumPyro/JAX at all. `GFA` works with just the base `cca-zoo` install.

```python
from cca_zoo.probabilistic import GFA

# latent_dimensions is an upper bound; drop_k=True (default) prunes it
model = GFA(latent_dimensions=5, random_state=0)
model.fit([X1, X2])

print(model.n_components_)  # <= 5: how many components survived pruning
print(model.view_relevance_)  # (n_views, n_components_) posterior mean alpha
```

`view_relevance_[i, k]` large means "dimension k is shrunk away in view i" — a component with a
small value in one view and a huge one in every other view is private to that view; a component
with small values everywhere is shared.

!!! warning "Convergence is a best-effort heuristic, not a guarantee"
    Fitting stops once a proxy (relative change in $z$) stays small for 1000 consecutive
    iterations, rather than the R package's full variational lower bound (which is provably
    monotonic, and so immune to this). Checking against a run with early stopping disabled
    caught this proxy staying below tolerance for 700+ iterations in the middle of a slow
    pruning process before rising again — a patience window makes this less likely, but can't
    rule it out. If `n_components_` looks larger than you'd expect, raise `max_iter` (default
    10000) rather than assuming the result is final.

`GFA.transform` and `GFA.weights` behave identically to the other two classes; `n_iter_` reports
how many iterations were actually run.

---

## `ProbabilisticCCA`: full MCMC

Inference is performed using the **No-U-Turn Sampler** (NUTS) via [NumPyro](https://num.pyro.ai/),
giving samples from the exact posterior (up to MCMC error). This is the more accurate option, but
full-batch NUTS scales poorly with $n$.

```python
from cca_zoo.probabilistic import ProbabilisticCCA

model = ProbabilisticCCA(
    latent_dimensions=2,
    center=True,
    num_warmup=500,
    num_samples=1000,
    random_state=0,
)
model.fit([X1, X2])
```

After fitting, `model.weights` holds the **posterior mean** loading matrices, and
`model.posterior_samples_` holds the full set of MCMC draws.

!!! note "Rotational symmetry"
    This model has an exact symmetry — $z \to zR$, $W_i \to W_i R$ for any orthogonal $R$
    shared across views leaves the likelihood unchanged — and different NUTS draws can settle
    on different rotations along that ridge. Averaging un-aligned draws would then be *biased
    toward zero* (draws pointing along different rotations partially cancel), so `fit` aligns
    every draw's loadings (and that draw's own `z`, to stay internally consistent) to a common
    reference via generalized Procrustes analysis before computing `weights_` or storing
    `posterior_samples_`. On a synthetic check, this raised a rotation-invariant coherence
    ratio (`||mean(W)||²` vs the mean of `||W||²` across draws — 1.0 if every draw agrees on a
    rotation) from 0.81 to 0.99.

### Transform (posterior mean prediction)

The latent representation is computed via the analytical posterior mean, using the posterior
mean loadings and noise variances:

$$
\Sigma_z = \left(I + \sum_i W_i^\top \Psi_i^{-1} W_i\right)^{-1}
$$

$$
\hat{\mathbf{z}}_j = \Sigma_z \sum_i W_i^\top \Psi_i^{-1} \mathbf{x}_{ij}
$$

```python
z = model.transform(
    [X1, X2]
)  # list with one array of shape (n_samples, latent_dimensions)
```

---

## `VariationalBayesCCA`: variational inference with automatic relevance determination

`VariationalBayesCCA` fits the same model, extended with a hierarchical **automatic relevance
determination (ARD)** prior shared across views:

$$
\alpha_k \sim \mathrm{Gamma}(a_0, b_0), \qquad W_i[:, k] \sim \mathcal{N}(0,\ \alpha_k^{-1} I)
$$

Because $\alpha_k$ ties every view's $k$-th loading column together, a shared latent dimension
is only retained if some view actually uses it — irrelevant dimensions get shrunk toward zero in
every view at once. The posterior mean of $\alpha_k$ (`model.ard_relevance_`) is a direct
usefulness score per dimension: large values mean "shrunk away, safe to drop". This gives
automatic latent-dimensionality selection, as an alternative to sweeping `latent_dimensions` with
`GridSearchCV`.

Inference uses mean-field **stochastic variational inference (SVI)** rather than the closed-form
conjugate updates in Wang (2007)'s original derivation — SVI reuses the same NumPyro generative
model as `ProbabilisticCCA` and extends to non-conjugate variants unmodified, at the cost of the
mean-field independence assumption between latent variables. It is substantially cheaper than
full NUTS.

```python
from cca_zoo.probabilistic import VariationalBayesCCA

# latent_dimensions is an upper bound here — set it generously and let ARD prune it
model = VariationalBayesCCA(
    latent_dimensions=5,
    num_steps=2000,
    learning_rate=1e-2,
    random_state=0,
)
model.fit([X1, X2])

print(model.ard_relevance_)  # one score per dimension; large = pruned
```

`model.transform` and `model.weights` behave identically to `ProbabilisticCCA`. `model.losses_`
holds the ELBO trace across SVI steps, useful for checking convergence.

---

## Full example

```python
import numpy as np
from cca_zoo.datasets import JointData
from cca_zoo.probabilistic import GFA, ProbabilisticCCA, VariationalBayesCCA

# Simulate correlated views
data = JointData(
    n_views=2,
    n_samples=100,
    n_features=[10, 10],
    latent_dimensions=2,
    signal_to_noise=3.0,
    random_state=0,
)
views = data.sample()

# Fit with GFA (no extra dependencies), requesting more dimensions than
# needed to see per-view ARD prune the unsupported ones
gfa_model = GFA(latent_dimensions=4, random_state=42)
gfa_model.fit(views)
print("GFA n_components_ after pruning:", gfa_model.n_components_)
print("Per-view relevance:", gfa_model.view_relevance_)

# Fit with MCMC (reduce warmup/samples for speed in examples)
mcmc_model = ProbabilisticCCA(
    latent_dimensions=2,
    num_warmup=200,
    num_samples=500,
    random_state=42,
)
mcmc_model.fit(views)
print("Posterior mean weights shape:", mcmc_model.weights[0].shape)  # (10, 2)

# Fit with variational inference, requesting more dimensions than needed
# to see ARD prune the unsupported ones
vb_model = VariationalBayesCCA(
    latent_dimensions=4,
    num_steps=2000,
    random_state=42,
)
vb_model.fit(views)
print("ARD relevance per dimension:", vb_model.ard_relevance_)

z = vb_model.transform(views)
print("Latent shape:", z[0].shape)  # (100, 4)
```

---

## Tips

- **Choosing a backend.** Start with `GFA` — no extra install, and per-view ARD is the more
  informative dimensionality signal if you care about shared-vs-private structure. Prefer
  `VariationalBayesCCA` when you specifically want ARD shared across views (fewer, more
  conservative retained dimensions) or need NumPyro's ecosystem. Prefer `ProbabilisticCCA` when
  you need the most accurate posterior (e.g. for final reported credible intervals) and $n$ is
  small enough for MCMC to be practical.
- **Warmup vs samples (MCMC).** NUTS requires a warm-up phase to adapt the step size. A typical
  setting is `num_warmup=500, num_samples=1000`. For exploration, `num_warmup=100,
  num_samples=200` is enough.
- **num_steps vs learning_rate (VB).** Check `model.losses_` — if it hasn't plateaued, increase
  `num_steps`. If it's noisy or diverging, lower `learning_rate`.
- **max_iter vs tol (GFA).** Convergence-based early stopping is a best-effort heuristic (see the
  warning above) — if `n_components_` looks too large, raise `max_iter` rather than lowering
  `tol` further.
- **Small datasets.** Probabilistic CCA is most useful when $n$ is small enough that
  uncertainty in the weights is meaningful (rough guide: $n < 500$ for MCMC; VB scales further).
- **Feature scaling.** Center and scale your views before fitting (`center=True` is the
  default). The priors on $W_i$ assume unit-scale inputs.
- **Convergence diagnostics.** Use [ArviZ](https://python.arviz.org/) on the NumPyro MCMC object
  (accessible via `model.mcmc_` on `ProbabilisticCCA`) for R-hat and effective sample size checks.
- **Comparing models.** Use `model.log_likelihood(held_out_views)` rather than `model.score(...)`
  when the question is "which model/latent_dimensions fits this data better" — it's the
  statistically proper Bayesian criterion, unlike the correlation-based `score` every model
  shares for `GridSearchCV` consistency.
