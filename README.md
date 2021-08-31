[![DOI](https://zenodo.org/badge/303801602.svg)](https://zenodo.org/badge/latestdoi/303801602)
[![codecov](https://codecov.io/gh/jameschapman19/cca_zoo/branch/master/graph/badge.svg?token=JHG9VUB0L8)](https://codecov.io/gh/jameschapman19/cca_zoo)
![Build Status](https://github.com/jameschapman19/cca_zoo/actions/workflows/python-package.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/cca-zoo/badge/?version=latest)](https://cca-zoo.readthedocs.io/en/latest/?badge=latest)
[![version](https://img.shields.io/pypi/v/cca-zoo)](https://pypi.org/project/cca-zoo/)
[![downloads](https://img.shields.io/pypi/dm/cca-zoo)](https://pypi.org/project/cca-zoo/)
[![Scrutinizer Code Quality](https://scrutinizer-ci.com/g/jameschapman19/cca_zoo/badges/quality-score.png?b=master)](https://scrutinizer-ci.com/g/jameschapman19/cca_zoo/?branch=master)

# Installation
Note: for standard install use: 
pip install cca-zoo

For deep learning elements use:
pip install cca-zoo[deep]

For probabilistic elements use:
pip install cca-zoo[probabilistic]

This means that there is no need to install the large pytorch package or numpyro to run cca-zoo unless you wish to use deep learning

# Documentation
Available at https://cca-zoo.readthedocs.io/en/latest/
  
# Citation:
If this repository was helpful to you please do give a star.

In case this work is used as part of research I attach a DOI bibtex entry:

```bibtex
@software{james_chapman_2021_4925892,
  author       = {James Chapman and
                  Hao-Ting Wang},
  title        = {jameschapman19/cca\_zoo:},
  month        = jun,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v1.6.1},
  doi          = {10.5281/zenodo.4925892},
  url          = {https://doi.org/10.5281/zenodo.4925892}
}
```

# Implemented Methods
## Standard Install
### CCA (Canonical Correlation Analysis)
Solutions based on either alternating least squares or as the solution to genrralized eigenvalue problem
### PLS (Partial Least Squares)

### rCCA (Ridge Regularized Canonical Correlation Analysis)
https://www.sciencedirect.com/science/article/abs/pii/0304407676900105?via%3Dihub
### GCCA (Generalized CCA)  :
https://academic.oup.com/biomet/article-abstract/58/3/433/233349?redirectedFrom=fulltext
### MCCA (Multiset CCA) : 
### K(M)CCA (kernel Multiset CCA) : 
### TCCA (Tensor CCA) :
https://arxiv.org/pdf/1502.02330.pdf
### KTCCA (kernel Tensor CCA) :
https://arxiv.org/pdf/1502.02330.pdf
### SCCA (Sparse CCA) :
https://onlinelibrary.wiley.com/doi/abs/10.1111/biom.13043
### SPLS (Sparse PLS/Penalized Matrix Decomposition)  :
https://web.stanford.edu/~hastie/Papers/PMD_Witten.pdf
### ElasticCCA (Penalized CCA) :
https://pubmed.ncbi.nlm.nih.gov/19689958/
### SWCCA (Sparse Weighted CCA) : 
https://arxiv.org/abs/1710.04792v1#:~:text=However%2C%20classical%20and%20sparse%20CCA%20models%20consider%20the,where%20weights%20are%20used%20for%20regularizing%20different%20samples.
### SpanCCA
http://akyrillidis.github.io/pubs/Conferences/cca.pdf

## Deep Install
### DCCA (Deep CCA) : 
https://ttic.uchicago.edu/~klivescu/papers/andrew_icml2013.pdf
https://arxiv.org/pdf/1510.02054v1.pdf
Using either Andrew's original Tracenorm Objective or Wang's alternating least squares solution
### DGCCA (Deep Generalized CCA)  :
https://www.aclweb.org/anthology/W19-4301.pdf
An alternative objective based on the linear GCCA solution. Can be extended to more than 2 views
### DMCCA (Deep Multiset CCA) :
https://arxiv.org/abs/1904.01775
An alternative objective based on the linear MCCA solution. Can be extended to more than 2 views
### DTCCA (Deep Tensor CCA) :
https://arxiv.org/pdf/2005.11914.pdf
### DCCAE (Deep Canonically Correlated Autoencoders) :
http://proceedings.mlr.press/v37/wangb15.pdf
### DVCCA/DVCCA Private (Deep variational CCA):
https://arxiv.org/pdf/1610.03454.pdf
## Probabilistic Install
### Variational Bayes CCA
https://ieeexplore.ieee.org/document/4182407

# Contributions
A guide to contributions is available at https://cca-zoo.readthedocs.io/en/latest/developer_info/contribute.html

# Sources

I've added this section to give due credit to the repositories that helped me in addition to their copyright notices in
the code where relevant.

Models can be tested on data from MNIST datasets provided by the torch package (https://pytorch.org/) and the UCI
dataset provided by mvlearn package (https://mvlearn.github.io/)

## Other Implementations of (regularised)CCA/PLS:

MATLAB implementation https://github.com/anaston/PLS_CCA_framework

## Implementation of Sparse PLS:

MATLAB implementation of SPLS by @jmmonteiro (https://github.com/jmmonteiro/spls)

## Other Implementations of DCCA/DCCAE:

Keras implementation of DCCA from @VahidooX's github page(https://github.com/VahidooX)
The following are the other implementations of DCCA in MATLAB and C++. These codes are written by the authors of the original paper:

Torch implementation of DCCA from @MichaelVll & @Arminarj: https://github.com/Michaelvll/DeepCCA

C++ implementation of DCCA from Galen Andrew's website (https://homes.cs.washington.edu/~galen/)

MATLAB implementation of DCCA/DCCAE from Weiran Wang's website (http://ttic.uchicago.edu/~wwang5/dccae.html)

MATLAB implementation of TCCA from https://github.com/rciszek/mdr_tcca

## Implementation of VAE:

Torch implementation of VAE (https://github.com/pytorch/examples/tree/master/vae)
