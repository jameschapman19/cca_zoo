# Probabilistic CCA

The `cca_zoo.probabilistic` module provides a Bayesian treatment of CCA via Markov Chain Monte
Carlo (MCMC) inference. It requires the `[probabilistic]` extra:

```bash
pip install cca-zoo[probabilistic]
```

---

## Background

Classical CCA finds a single point estimate of the canonical weights. **Probabilistic CCA**
(Bach & Jordan 2005; Wang 2007) instead defines a generative model with explicit priors and
uses MCMC to obtain a full posterior distribution over the weights, enabling uncertainty
quantification.

The generative model is:

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

Inference is performed using the **No-U-Turn Sampler** (NUTS) via
[NumPyro](https://num.pyro.ai/).

---

## Usage

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

After fitting, `model.weights` holds the **posterior mean** loading matrices.

### Transform (posterior mean prediction)

The latent representation is computed via the analytical posterior mean:

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

## Full example

```python
import numpy as np
from cca_zoo.datasets import JointData
from cca_zoo.probabilistic import ProbabilisticCCA

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
model = ProbabilisticCCA(
    latent_dimensions=2,
    num_warmup=200,
    num_samples=500,
    random_state=42,
)
model.fit(views)

print("Posterior mean weights shape:", model.weights[0].shape)  # (10, 2)

z = model.transform(views)
print("Latent shape:", z[0].shape)  # (100, 2)
```

---

## Tips

- **Warmup vs samples.** NUTS requires a warm-up phase to adapt the step size. A typical
  setting is `num_warmup=500, num_samples=1000`. For exploration, `num_warmup=100,
  num_samples=200` is enough.
- **Small datasets.** Probabilistic CCA is most useful when $n$ is small enough that
  uncertainty in the weights is meaningful (rough guide: $n < 500$).
- **Feature scaling.** Center and scale your views before fitting (`center=True` is the
  default). The prior on $W_i$ assumes unit-scale inputs.
- **Convergence diagnostics.** Use [ArviZ](https://python.arviz.org/) on the NumPyro
  MCMC object (accessible via `model.mcmc_`) for R-hat and effective sample size checks.
