<div align="center">

<img src="docs/source/cca-zoo-logo.jpg" alt="drawing" width="200"/>

# CCA-Zoo

**Unlock the hidden relationships in multiview data.**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5748062.svg)](https://doi.org/10.5281/zenodo.4382739)
[![codecov](https://codecov.io/gh/jameschapman19/cca_zoo/branch/main/graph/badge.svg?token=JHG9VUB0L8)](https://codecov.io/gh/jameschapman19/cca_zoo)
![Build Status](https://github.com/jameschapman19/cca_zoo/actions/workflows/changes.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/cca-zoo/badge/?version=latest)](https://cca-zoo.readthedocs.io/en/latest/?badge=latest)
[![version](https://img.shields.io/pypi/v/cca-zoo)](https://pypi.org/project/cca-zoo/)
[![downloads](https://img.shields.io/pypi/dm/cca-zoo)](https://pypi.org/project/cca-zoo/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.03823/status.svg)](https://doi.org/10.21105/joss.03823)


</div>

In an era inundated with data, unraveling the intricate connections between different data views is paramount. **CCA-Zoo** emerges as the definitive toolkit for this, offering an extensive suite of linear, kernel, and deep methods for canonical correlation analysis. 

Inspired by the simplicity and effectiveness of `scikit-learn` and `mvlearn`, CCA-Zoo offers a seamless experience with its `fit`/`transform`/`fit_transform` methods.

## Table of Contents

- [Installation](#installation)
- [Documentation](#documentation)
- [Citation](#citation)
- [Contributions](#contributions)
- [Sources](#sources)

## 🚀 Get Started in 5 Minutes

### Installation

Whether you're a `pip` enthusiast or a `poetry` lover, we've got you covered:

```bash
pip install cca-zoo
# or for the adventurous
pip install cca-zoo[probabilistic]
```

Poetry aficionados can dive in with:

```bash
poetry add cca-zoo
# or with a twist of probability
poetry add cca-zoo[probabilistic]
```


## 📚 Dive Deep

Embark on a journey through multiview correlations with our [comprehensive guide](https://cca-zoo.readthedocs.io/en/latest/).

## 🙏 Show Your Love

CCA-Zoo thrives on the support of its community. If it's made a splash in your research, consider sprinkling some stardust by citing our JOSS paper or simply starring our repo. Every gesture counts!

📜 Chapman et al., (2021). CCA-Zoo: A collection of Regularized, Deep Learning based, Kernel, and Probabilistic CCA methods in a scikit-learn style framework. Journal of Open Source Software, 6(68), 3823, [Link](https://doi.org/10.21105/joss.03823).

## 👩‍💻 Contribute

Every idea, every line of code adds value. Check out our [contribution guide](https://cca-zoo.readthedocs.io/en/latest/developer_info/contribute.html) and help CCA-Zoo soar to new heights!

## 🙌 Props

A nod to the stalwarts and pioneers whose work paved the way. Dive into their implementations and explorations:

- Regularised CCA/PLS: [MATLAB](https://github.com/anaston/PLS_CCA_framework)
- Sparse PLS: [MATLAB SPLS](https://github.com/jmmonteiro/spls)
- DCCA/DCCAE: [Keras DCCA](https://github.com/VahidooX), [Torch DCCA](https://github.com/Michaelvll/DeepCCA), and more...
- VAE: [Torch VAE](https://github.com/pytorch/examples/tree/master/vae)
