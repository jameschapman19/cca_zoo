[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5748062.svg)](https://doi.org/10.5281/zenodo.4382739)
[![codecov](https://codecov.io/gh/jameschapman19/cca_zoo/branch/main/graph/badge.svg?token=JHG9VUB0L8)](https://codecov.io/gh/jameschapman19/cca_zoo)
![Build Status](https://github.com/jameschapman19/cca_zoo/actions/workflows/python-package.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/cca-zoo/badge/?version=latest)](https://cca-zoo.readthedocs.io/en/latest/?badge=latest)
[![version](https://img.shields.io/pypi/v/cca-zoo)](https://pypi.org/project/cca-zoo/)
[![downloads](https://img.shields.io/pypi/dm/cca-zoo)](https://pypi.org/project/cca-zoo/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.03823/status.svg)](https://doi.org/10.21105/joss.03823)

# CCA-Zoo

`cca-zoo` is a collection of linear, kernel, and deep methods for canonical correlation analysis of multiview data. 
Where possible it follows the `scikit-learn`/`mvlearn` APIs and models therefore have `fit`/`transform`/`fit_transform` methods as standard.

## Installation

Dependency of some implemented algorithms are heavy, such as `pytorch` and `numpyro`. 
We provide several options to accomodate the user's needs.
For full details of algorithms included, please refer to section [Implemented Methods](#implemented-methods)

Standard installation: 

```
pip install cca-zoo
```
For deep learning elements use:
```
pip install cca-zoo[deep]
```

For probabilistic elements use:
```
pip install cca-zoo[probabilistic]
```
## Documentation
Available at https://cca-zoo.readthedocs.io/en/latest/
  
## Citation:
CCA-Zoo is intended as research software. Citations and use of our software help us justify the effort which has gone into, and will keep going into, maintaining and growing this project. Stars on the repo are also greatly appreciated :)

If you have used CCA-Zoo in your research, please consider citing our JOSS paper:

Chapman et al., (2021). CCA-Zoo: A collection of Regularized, Deep Learning based, Kernel, and Probabilistic CCA methods in a scikit-learn style framework. Journal of Open Source Software, 6(68), 3823, https://doi.org/10.21105/joss.03823

With bibtex entry:

```bibtex
@article{Chapman2021,
  doi = {10.21105/joss.03823},
  url = {https://doi.org/10.21105/joss.03823},
  year = {2021},
  publisher = {The Open Journal},
  volume = {6},
  number = {68},
  pages = {3823},
  author = {James Chapman and Hao-Ting Wang},
  title = {CCA-Zoo: A collection of Regularized, Deep Learning based, Kernel, and Probabilistic CCA methods in a scikit-learn style framework},
  journal = {Journal of Open Source Software}
}
```

## Implemented Methods

### Standard Install
- CCA (Canonical Correlation Analysis): Solutions based on either alternating least squares or as the solution to genrralized eigenvalue problem
- PLS (Partial Least Squares)
- [rCCA (Ridge Regularized Canonical Correlation Analysis)](https://www.sciencedirect.com/science/article/abs/pii/0304407676900105?via%3Dihub)
- [GCCA (Generalized CCA)](https://academic.oup.com/biomet/article-abstract/58/3/433/233349?redirectedFrom=fulltext)
- MCCA (Multiset CCA)
- K(M)CCA (kernel Multiset CCA)
- [TCCA (Tensor CCA)](https://arxiv.org/pdf/1502.02330.pdf)
- [KTCCA (kernel Tensor CCA)](https://arxiv.org/pdf/1502.02330.pdf)
- [SCCA (Sparse CCA)](https://onlinelibrary.wiley.com/doi/abs/10.1111/biom.13043)
- [SPLS (Sparse PLS/Penalized Matrix Decomposition](https://web.stanford.edu/~hastie/Papers/PMD_Witten.pdf)
- [ElasticCCA (Penalized CCA)](https://pubmed.ncbi.nlm.nih.gov/19689958/)
- [SWCCA (Sparse Weighted CCA)](https://arxiv.org/abs/1710.04792v1#:~:text=However%2C%20classical%20and%20sparse%20CCA%20models%20consider%20the,where%20weights%20are%20used%20for%20regularizing%20different%20samples)
- [SpanCCA](http://akyrillidis.github.io/pubs/Conferences/cca.pdf)

### `[deep]` Install
- DCCA (Deep CCA)

  Using either Andrew's original [Tracenorm Objective](https://ttic.uchicago.edu/~klivescu/papers/andrew_icml2013.pdf) or Wang's [alternating least squares solution](https://arxiv.org/pdf/1510.02054v1.pdf)
  
- [DGCCA (Deep Generalized CCA)](https://www.aclweb.org/anthology/W19-4301.pdf)

  An alternative objective based on the linear GCCA solution. Can be extended to more than 2 views
 
- [DMCCA (Deep Multiset CCA)](https://arxiv.org/abs/1904.01775)

  An alternative objective based on the linear MCCA solution. Can be extended to more than 2 views
  
- [DTCCA (Deep Tensor CCA)](https://arxiv.org/pdf/2005.11914.pdf)
- [DCCAE (Deep Canonically Correlated Autoencoders)](http://proceedings.mlr.press/v37/wangb15.pdf)
- [DVCCA/DVCCA Private (Deep variational CCA)](https://arxiv.org/pdf/1610.03454.pdf)

### `[probabilistic]` Install
- [Variational Bayes CCA](https://ieeexplore.ieee.org/document/4182407)

## Contributions
A guide to contributions is available at https://cca-zoo.readthedocs.io/en/latest/developer_info/contribute.html

## Sources

I've added this section to give due credit to the repositories that helped me in addition to their copyright notices in
the code where relevant.

### Other Implementations of (regularised)CCA/PLS

[MATLAB implementation](https://github.com/anaston/PLS_CCA_framework)

### Implementation of Sparse PLS

MATLAB implementation of SPLS by [@jmmonteiro](https://github.com/jmmonteiro/spls)

### Other Implementations of DCCA/DCCAE

Keras implementation of DCCA from [@VahidooX's github page](https://github.com/VahidooX)

The following are the other implementations of DCCA in MATLAB and C++. These codes are written by the authors of the original paper:

[Torch implementation](https://github.com/Michaelvll/DeepCCA) of DCCA from @MichaelVll & @Arminarj

C++ implementation of DCCA from Galen Andrew's [website](https://homes.cs.washington.edu/~galen/)

MATLAB implementation of DCCA/DCCAE from Weiran Wang's [website](http://ttic.uchicago.edu/~wwang5/dccae.html)

MATLAB implementation of [TCCA](https://github.com/rciszek/mdr_tcca)

### Implementation of VAE

[Torch implementation of VAE](https://github.com/pytorch/examples/tree/master/vae)
