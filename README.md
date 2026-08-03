<div align="center">
<img src="docs/logos/cca-zoo-logo.svg" alt="CCA-Zoo" width="180"/>

# CCA-Zoo

**Multiview Canonical Correlation Analysis in Python**

[![PyPI](https://img.shields.io/pypi/v/cca-zoo)](https://pypi.org/project/cca-zoo/)
[![Python](https://img.shields.io/pypi/pyversions/cca-zoo)](https://pypi.org/project/cca-zoo/)
[![CI](https://github.com/jameschapman19/cca_zoo/actions/workflows/ci.yml/badge.svg)](https://github.com/jameschapman19/cca_zoo/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jameschapman19/cca_zoo/branch/main/graph/badge.svg)](https://codecov.io/gh/jameschapman19/cca_zoo)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.03823/status.svg)](https://doi.org/10.21105/joss.03823)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Types: mypy strict](https://img.shields.io/badge/types-mypy%20strict-blue.svg)](https://mypy-lang.org/)

</div>

CCA-Zoo is a Python library of **reference implementations of Canonical Correlation Analysis
(CCA) algorithms from the literature**, from classical CCA (Hotelling 1936) through sparse,
kernel, deep, and probabilistic variants — each documented with the paper it comes from. It's
also built to be used directly: every model follows the same
[scikit-learn](https://scikit-learn.org) estimator API (`fit`, `transform`, `fit_transform`,
`score`), is fully typed (PEP 561), and is tested against known closed-form solutions where one
exists.

---

## Installation

```bash
uv add cca-zoo        # or: pip install cca-zoo
```

Install optional extras as needed:

```bash
uv add "cca-zoo[deep]"          # DCCA variants (requires PyTorch + Lightning)
uv add "cca-zoo[probabilistic]" # Probabilistic CCA (requires NumPyro + JAX)
uv add "cca-zoo[tree]"          # TreeCCA (requires XGBoost, optionally LightGBM)
uv add "cca-zoo[all]"           # Everything above
```

(substitute `pip install` for `uv add` if you're not using [uv](https://docs.astral.sh/uv/))

---

## Quick start

```python
from cca_zoo.datasets import JointData
from cca_zoo.linear import CCA

# Generate correlated two-view data from a linear latent variable model
data = JointData(
    n_views=2,
    n_samples=200,
    n_features=[50, 50],
    latent_dimensions=2,
    signal_to_noise=2.0,
    random_state=0,
)
train_views = data.sample()
test_views = data.sample()

# Fit CCA and evaluate
model = CCA(latent_dimensions=2).fit(train_views)
print(model.score(test_views))  # canonical correlations, shape (2,)

# Project views into the shared latent space
z1, z2 = model.transform(test_views)  # each shape (200, 2)
```

---

## Available methods

### `cca_zoo.linear`

| Class | Description | Views |
|---|---|---|
| `CCA` | Standard CCA (Hotelling 1936) | 2 |
| `rCCA` | Regularised CCA / canonical ridge | 2 |
| `PLS` | Partial Least Squares | 2 |
| `MCCA` | Multiset CCA — pairwise sum objective | ≥2 |
| `GCCA` | Generalised CCA — shared latent projection | ≥2 |
| `TCCA` | Tensor CCA — higher-order cross-moment | ≥2 |
| `PartialCCA` | CCA adjusted for confounding variables (Rao 1969) | ≥2 |
| `GRCCA` | Group-regularised CCA (Tuzhilina, Tozzi & Hastie 2021) | ≥2 |
| `CCAR3` | CCA via reduced-rank regression, row-sparse in high dimensions (Donnat & Tuzhilina 2024) | 2 |
| `CCA_EY` | Stochastic Eckart-Young CCA (unconstrained gradient descent) | 2 |
| `PLS_EY` | Stochastic Eckart-Young PLS (unconstrained gradient descent) | 2 |
| `MCCA_EY` | Multiview Eckart-Young CCA (unconstrained gradient descent) | ≥2 |
| `SCCA_PMD` | Sparse CCA via PMD (Witten 2009) | ≥2 |
| `SCCA_ADMM` | Sparse CCA via ADMM (Suo 2017) | ≥2 |
| `SCCA_IPLS` | Sparse CCA via iterative PLS (Mai & Zhang 2019) | ≥2 |
| `SCCA_Span` | SpanCCA (Asteris 2016) | ≥2 |
| `ElasticCCA` | Elastic net regularised CCA (Waaijenborg 2008) | ≥2 |
| `ParkhomenkoCCA` | Soft-threshold sparse CCA (Parkhomenko 2009) | ≥2 |
| `PLS_ALS` | ALS variant of PLS (power iteration) | ≥2 |

### `cca_zoo.nonparametric`

| Class | Description |
|---|---|
| `KCCA` | Kernel CCA |
| `KGCCA` | Kernel Generalised CCA |
| `KTCCA` | Kernel Tensor CCA |

### `cca_zoo.tree` *(requires `[tree]`)*

| Class | Description | Views |
|---|---|---|
| `TreeCCA` | Gradient-boosted-tree CCA (Eckart-Young objective) | ≥2 |

### `cca_zoo.deep` *(requires `[deep]`)*

Built on PyTorch Lightning — models are trained with a standard `lightning.Trainer`, not a
`fit()` wrapper. See the [deep learning guide](https://jameschapman19.github.io/cca_zoo/user-guide/deep/).

| Class | Reference |
|---|---|
| `DCCA` | Andrew et al. 2013 — pluggable objective |
| `DCCA_EY` | Eigengame / Eckart-Young objective |
| `DCCA_NOI` | Wang et al. 2015 — non-linear orthogonal iterations |
| `DCCA_SDL` | Chang et al. 2018 — stochastic decorrelation loss |
| `DCCAE` | Wang et al. 2015 — with autoencoder reconstruction |
| `DVCCA` | Wang et al. 2016 — variational |
| `DTCCA` | Wong et al. 2021 — deep tensor CCA |
| `DMCCA` | Deep multiset CCA — pairwise-sum objective, ≥2 views |
| `DGCCA` | Benton et al. 2019 — deep generalised CCA, ≥2 views |
| `SplitAE` | Split autoencoder baseline |
| `BarlowTwins` | Zbontar et al. 2021 |
| `VICReg` | Bardes et al. 2022 |

### `cca_zoo.probabilistic` *(requires `[probabilistic]`)*

| Class | Reference |
|---|---|
| `ProbabilisticCCA` | Bach & Jordan 2005; Wang 2007 — MCMC via NumPyro |

### `cca_zoo.model_selection`

| Class | Description |
|---|---|
| `GridSearchCV` | Cross-validated hyperparameter search for multiview models |

---

## Documentation

Full documentation, user guides, and API reference at:
**[https://jameschapman19.github.io/cca_zoo/](https://jameschapman19.github.io/cca_zoo/)**

---

## Citing

If CCA-Zoo is useful in your research, please cite:

```bibtex
@article{Chapman2021,
  title   = {{CCA-Zoo}: A collection of Regularized, Deep Learning based, Kernel,
             and Probabilistic {CCA} methods in a scikit-learn style framework},
  author  = {Chapman, James and Wang, Hao-Ting and Wells, Lennie and Wiesner, Johannes},
  journal = {Journal of Open Source Software},
  volume  = {6},
  number  = {68},
  pages   = {3823},
  year    = {2021},
  doi     = {10.21105/joss.03823},
}
```

---

## Contributing

Contributions are welcome. See [docs/contributing.md](docs/contributing.md) for development setup, coding standards, and pull request guidelines.
