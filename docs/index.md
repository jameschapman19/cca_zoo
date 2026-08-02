---
hide:
  - toc
---

[![PyPI](https://img.shields.io/pypi/v/cca-zoo)](https://pypi.org/project/cca-zoo/)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/cca-zoo)](https://pypi.org/project/cca-zoo/)
[![MIT License](https://img.shields.io/badge/license-MIT-green)](https://github.com/jameschapman19/cca_zoo/blob/main/LICENSE)
[![CI](https://github.com/jameschapman19/cca_zoo/actions/workflows/ci.yml/badge.svg)](https://github.com/jameschapman19/cca_zoo/actions/workflows/ci.yml)

# CCA-Zoo

**Multiview Canonical Correlation Analysis for Python —
scikit-learn compatible, research-grade, batteries included.**

```bash
pip install cca-zoo
```

[Get Started](getting-started.md){ .md-button .md-button--primary }
[API Reference](api/linear.md){ .md-button }

---

## What is CCA?

Given two or more views of the same observations — brain imaging and behavioural scores,
gene expression and phenotypic data, audio and video features — **Canonical Correlation
Analysis** finds projections that maximise correlation between the projected views.

CCA-Zoo extends classical CCA in several directions:

<div class="grid cards" markdown>

- :material-chart-scatter-plot: **Linear & regularised**

    Classical CCA, ridge-regularised rCCA, PLS, and seven sparse/elastic-net
    variants for high-dimensional settings.

    [Linear methods →](user-guide/linear.md)

- :material-vector-curve: **Kernel & nonparametric**

    KCCA, KGCCA, and KTCCA bring nonlinear relationships into reach via the
    kernel trick — no explicit feature map needed.

    [Kernel methods →](user-guide/nonparametric.md)

- :material-pine-tree: **Gradient-boosted trees**

    TreeCCA fits nonlinear encoders as gradient-boosted-tree ensembles, with
    built-in per-component feature importance — no SHAP required.

    [Tree methods →](user-guide/tree.md)

- :material-brain: **Deep learning**

    DCCA and variants (EY, NOI, SDL, DCCAE, DVCCA, DTCCA, BarlowTwins, VICReg)
    using your own `nn.Module` encoders with PyTorch Lightning.

    [Deep methods →](user-guide/deep.md)

- :material-chart-bell-curve: **Probabilistic**

    Full Bayesian treatment of CCA via NUTS MCMC with NumPyro — posterior
    inference over latent variables and loadings.

    [Probabilistic →](user-guide/probabilistic.md)

</div>

---

## Unified API

Every model follows the same three-step scikit-learn pattern:

```python
from cca_zoo.linear import CCA, rCCA, PLS
from cca_zoo.nonparametric import KCCA

# 1. construct
model = CCA(latent_dimensions=2)

# 2. fit — views is a list of arrays, one per dataset
model.fit([X1, X2])

# 3. use
z1, z2 = model.transform([X1, X2])
corrs  = model.score([X1, X2])    # canonical correlations, shape (2,)
W1, W2 = model.weights            # weight matrices
```

Models are `sklearn.base.BaseEstimator` subclasses, so they work directly with
`GridSearchCV`, `Pipeline`, and cross-validation utilities.

---

## Navigate the docs

<div class="grid cards" markdown>

- :material-rocket-launch: **[Getting Started](getting-started.md)**

    Installation, quick start examples, and core concepts.

- :material-book-open-variant: **[User Guide](user-guide/linear.md)**

    In-depth explanations of each method family with usage guidance.

- :material-code-tags: **[API Reference](api/linear.md)**

    Full class and method documentation auto-generated from source.

- :material-source-pull: **[Contributing](contributing.md)**

    Development setup, coding standards, and how to contribute.

</div>
