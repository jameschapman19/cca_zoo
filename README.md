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
