# Probabilistic CCA

The `cca_zoo.probabilistic` module provides a Bayesian treatment of CCA, with two inference
backends: full MCMC sampling (`ProbabilisticCCA`) and mean-field variational inference with
automatic latent-dimensionality selection (`VariationalBayesCCA`). It requires the
`[probabilistic]` extra:

```bash
pip install cca-zoo[probabilistic]
```

---

## Background

Classical CCA finds a single point estimate of the canonical weights. **Probabilistic CCA**
(Bach & Jordan 2005) instead defines a generative model with explicit priors, enabling
uncertainty quantification over the weights. The base generative model is:

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

Both classes share this model and the same posterior-mean projection formula for `transform`;
they differ only in how the posterior is approximated.

### Scoring: correlation vs likelihood

Both classes implement `score` (average pairwise correlation) for consistency with every other
model in `cca_zoo` — this is what `GridSearchCV` optimizes by default. But a correlation between
projections isn't the statistically natural fit criterion for a *probabilistic* model. Both
classes also implement `log_likelihood`, the marginal log-likelihood of the data with the shared
latent variable integrated out:

$$
\mathbf{x} \sim \mathcal{N}\!\left(0,\ \Psi + WW^\top\right)
$$

where $\mathbf{x}$ is the concatenation of every view's centred features for one sample, $W$
stacks every view's loading matrix, and $\Psi$ is the (block-)diagonal noise-variance matrix.
This is evaluated jointly across the concatenated views rather than per view: because every view
shares the same $z$, marginalising it induces cross-view covariance that a per-view likelihood
would silently ignore. Use `log_likelihood` to compare `latent_dimensions` choices, or to compare
`ProbabilisticCCA` against `VariationalBayesCCA` on the same data — larger (less negative) is
better.

```python
model.log_likelihood([X1, X2])  # mean log-likelihood per sample
```

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
from cca_zoo.probabilistic import ProbabilisticCCA, VariationalBayesCCA

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

- **Choosing a backend.** Prefer `VariationalBayesCCA` for exploration, larger $n$, or when you
  want automatic dimensionality selection via ARD. Prefer `ProbabilisticCCA` when you need the
  most accurate posterior (e.g. for final reported credible intervals) and $n$ is small enough
  for MCMC to be practical.
- **Warmup vs samples (MCMC).** NUTS requires a warm-up phase to adapt the step size. A typical
  setting is `num_warmup=500, num_samples=1000`. For exploration, `num_warmup=100,
  num_samples=200` is enough.
- **num_steps vs learning_rate (VB).** Check `model.losses_` — if it hasn't plateaued, increase
  `num_steps`. If it's noisy or diverging, lower `learning_rate`.
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
