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
uv add "cca-zoo[tree]"          # XGBoostCCA, LightGBMCCA, CatBoostCCA
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

| Class | Description | Citation | Views |
|---|---|---|---|
| `CCA` | Standard CCA | Hotelling (1936) | 2 |
| `rCCA` | Regularised CCA / canonical ridge | Vinod (1976) | 2 |
| `PLS` | Partial Least Squares | Wold (1975) | 2 |
| `MCCA` | Multiset CCA — pairwise sum objective | Kettenring (1971) | ≥2 |
| `GCCA` | Generalised CCA — shared latent projection | Tenenhaus & Tenenhaus (2011) | ≥2 |
| `TCCA` | Tensor CCA — higher-order cross-moment | Kim, Wong & Cipolla (2007) | ≥2 |
| `PartialCCA` | CCA adjusted for confounding variables | Rao (1969) | ≥2 |
| `GRCCA` | Group-regularised CCA | Tuzhilina, Tozzi & Hastie (2021) | ≥2 |
| `CCAR3` | CCA via reduced-rank regression, row-sparse in high dimensions | Donnat & Tuzhilina (2024) | 2 |
| `ECCA` | CCA via reduced-rank regression, entrywise-sparse (ccar3 package's `ecca`) | Donnat & Tuzhilina (2024) | 2 |
| `GraphicalLassoCCA` | MCCA with an L1-penalised sparse-precision within-view covariance | Friedman, Hastie & Tibshirani (2008) | ≥2 |
| `CCAEY` | Eckart-Young CCA, full-batch L-BFGS-B (2 or more views) | Chapman, Wells & Lawry Aguila (2024) | ≥2 |
| `PLSEY` | Eckart-Young PLS, full-batch L-BFGS-B | Chapman, Wells & Lawry Aguila (2024) | ≥2 |
| `HuberCCA` | Bounded-influence (Huber-style) EY-CCA, robust to high-leverage outliers | — | ≥2 |
| `RANSACCCA` | Robust CCA via random sample consensus, robust to mismatched/corrupted rows | — | ≥2 |
| `TrimmedCCA` | Robust CCA via LTS/MCD-style concentration steps, holds up near ~50% contamination | Rousseeuw & Van Driessen (1999) | ≥2 |

### `cca_zoo.nonparametric`

| Class | Description | Citation |
|---|---|---|
| `KCCA` | Kernel CCA | Hardoon, Szedmak & Shawe-Taylor (2004) |
| `KGCCA` | Kernel Generalised CCA | Tenenhaus, Philippe & Frouin (2015) |
| `KTCCA` | Kernel Tensor CCA | Kim, Wong & Cipolla (2007) |
| `ManifoldCCA` | Transductive CCA over a shared graph Laplacian or LLE operator | Belkin & Niyogi (2003); Roweis & Saul (2000) |

### `cca_zoo.tree` *(requires `[tree]`)*

| Class | Description | Citation | Views |
|---|---|---|---|
| `XGBoostCCA` | Gradient-boosted-tree CCA via XGBoost (Eckart-Young objective) | Chapman (2026) | ≥2 |
| `LightGBMCCA` | Gradient-boosted-tree CCA via LightGBM (Eckart-Young objective) | Chapman (2026) | ≥2 |
| `CatBoostCCA` | Gradient-boosted-tree CCA via CatBoost (Eckart-Young objective) | Chapman (2026) | ≥2 |

### `cca_zoo.gam`

| Class | Description | Citation | Views |
|---|---|---|---|
| `GAMCCA` | Generalized-additive-model CCA (Eckart-Young objective) | Chapman, Wells & Lawry Aguila (2024) | ≥2 |
| `MARSCCA` | Multivariate-adaptive-regression-spline CCA with optional within-view interactions (Eckart-Young objective) | Friedman (1991) | ≥2 |

### `cca_zoo.gp`

| Class | Description | Citation | Views |
|---|---|---|---|
| `GPCCA` | Gaussian-process CCA (Eckart-Young objective), with predictive uncertainty | Chapman, Wells & Lawry Aguila (2024) | ≥2 |

### `cca_zoo.sparse`

| Class | Description | Citation | Views |
|---|---|---|---|
| `ElasticNetCCA` | Sparse linear CCA via coordinate descent (Eckart-Young objective) | — | ≥2 |
| `MultiTaskElasticNetCCA` | `ElasticNetCCA` with row-group sparsity shared across latent dimensions | — | ≥2 |
| `OrthogonalMatchingPursuitCCA` | Fixed-cardinality sparse linear CCA via greedy selection (Eckart-Young objective) | — | ≥2 |
| `PMDCCA` | Sparse CCA via PMD | Witten, Tibshirani & Hastie (2009) | ≥2 |
| `ADMMCCA` | Sparse CCA via ADMM | Suo, Mineiro & Anandkumar (2017) | ≥2 |
| `IPLSCCA` | Sparse CCA via iterative PLS | Mai & Zhang (2019) | ≥2 |
| `SpanCCA` | Hard-threshold ALS inspired by the SpanCCA algorithm | Asteris, Kyrillidis, Koyejo & Poldrack (2016) | ≥2 |
| `WaijenborgCCA` | Elastic net regularised CCA | Waaijenborg, de Witt Hamer & Zwinderman (2008) | ≥2 |
| `ParkhomenkoCCA` | Soft-threshold sparse CCA | Parkhomenko, Tritchler & Beyene (2009) | ≥2 |
| `SAR` | Sparse alternating regression, BIC-selected penalty | Wilms & Croux (2015) | ≥2 |

### `cca_zoo.stochastic`

| Class | Description | Citation | Views |
|---|---|---|---|
| `StochasticCCAEY` | `CCAEY`, fit by mini-batch momentum SGD | Chapman, Wells & Lawry Aguila (2024) | ≥2 |

### `cca_zoo.deep` *(requires `[deep]`)*

Built on PyTorch Lightning — models are trained with a standard `lightning.Trainer`, not a
`fit()` wrapper. See the [deep learning guide](https://jameschapman19.github.io/cca_zoo/user-guide/deep/).

| Class | Description | Citation |
|---|---|---|
| `DCCA` | Deep CCA, pluggable objective | Andrew et al. (2013) |
| `DCCA_EY` | Deep CCA via Eigengame / Eckart-Young objective | Chapman, Wells & Lawry Aguila (2024) |
| `DCCA_NOI` | Deep CCA via non-linear orthogonal iterations | Wang et al. (2015) |
| `DCCA_SDL` | Deep CCA via stochastic decorrelation loss | Chang, Xiang & Hospedales (2018) |
| `DCCAE` | Deep CCA with autoencoder reconstruction | Wang et al. (2015) |
| `DVCCA` | Deep variational CCA | Wang et al. (2016) |
| `DTCCA` | Deep tensor CCA | Wong et al. (2021) |
| `DMCCA` | Deep multiset CCA — pairwise-sum objective, ≥2 views | Kettenring (1971) |
| `DGCCA` | Deep generalised CCA, ≥2 views | Benton et al. (2019) |
| `SplitAE` | Split autoencoder baseline | — |
| `BarlowTwins` | Self-supervised learning via redundancy reduction | Zbontar et al. (2021) |
| `VICReg` | Variance-Invariance-Covariance Regularization | Bardes, Ponce & LeCun (2022) |

### `cca_zoo.probabilistic`

| Class | Description | Citation |
|---|---|---|
| `GFA` | Group Factor Analysis, per-view ARD; no extra dependencies | Klami, Virtanen & Kaski (2013) |
| `ProbabilisticCCA` *(requires `[probabilistic]`)* | MCMC via NumPyro | Bach & Jordan (2005) |
| `VariationalBayesCCA` *(requires `[probabilistic]`)* | Variational inference + ARD via NumPyro | Wang (2007) |

### `cca_zoo.model_selection`

| Class | Description | Citation |
|---|---|---|
| `GridSearchCV` | Cross-validated hyperparameter search for multiview models | — |

---

## Documentation

Full documentation, user guides, and API reference at:
**[https://jameschapman19.github.io/cca_zoo/](https://jameschapman19.github.io/cca_zoo/)**

See [CHANGELOG.md](CHANGELOG.md) for what's changed between releases.

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

Contributions are welcome. See [docs/contributing.md](docs/contributing.md) for development setup, coding standards, and pull request guidelines. Please also read our [Code of Conduct](CODE_OF_CONDUCT.md).

Found a security issue? See [SECURITY.md](SECURITY.md) for how to report it privately.
