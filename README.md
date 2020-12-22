[![DOI](https://zenodo.org/badge/303801602.svg)](https://zenodo.org/badge/latestdoi/303801602)
[![codecov](https://codecov.io/gh/jameschapman19/cca_zoo/branch/master/graph/badge.svg?token=JHG9VUB0L8)](https://codecov.io/gh/jameschapman19/cca_zoo)
[![Build Status](https://www.travis-ci.com/jameschapman19/cca_zoo.svg?branch=master)](https://www.travis-ci.com/jameschapman19/cca_zoo)
[![Documentation Status](https://readthedocs.org/projects/cca-zoo/badge/?version=latest)](https://cca-zoo.readthedocs.io/en/latest/?badge=latest)

# Canonical Correlation Analysis Methods: cca-zoo
## Linear CCA/PLS:
A variety of linear CCA and PLS methods implemented where possible as the solutions to generalized eigenvalue problems and otherwise using alternating minimization methods for non-convex optimisation based on least squares
### CCA (Canonical Correlation Analysis)
Solutions based on either alternating least squares or as the solution to genrralized eigenvalue problem
### GCCA (Generalized CCA)  
### MCCA (Multiset CCA)
### SCCA (Sparse CCA) :
Mai's sparse CCA
### SPLS (Sparse PLS/Penalized Matrix Decomposition)  :
Witten's sparse CCA
### PCCA (Penalized CCA - elastic net)
Waiijenborg's elastic penalized CCA
## Deep CCA:
A variety of Deep CCA and related methods. All allow for user to pass their own model architectures. Recently added solutions to DCCA using nob-linear orthogonal iterations (or alternating least squares)
### DCCA (Deep CCA) : 
Using either Andrew's original Tracenorm Objective or Wang's alternating least squares solution
### DGCCA (Deep Generalized CCA)  :
An alternative objective based on the linear GCCA solution. Can be extended to more than 2 views
### DMCCA (Deep Multiset CCA) :
An alternative objective based on the linear MCCA solution. Can be extended to more than 2 views
### DCCAE (Deep Canonically Correlated Autoencoders)
### DVCCA/DVCCA Private (Deep variational CCA):
Wang's DVCCA and DVCCA Private
## Kernel CCA:
### Linear Kernel  
### RBF Kernel  
### Polynomial Kernels  


### Documentation at https://cca-zoo.readthedocs.io/en/latest/
### Can be installed using pip install cca-zoo
### Recently added DCCA by non-linear orthogonal iterations (https://arxiv.org/pdf/1510.02054.pdf)
  
# Credits:
If this repository was helpful please do give a star

In case this work is used as part of research I attach a bibtex entry:

```bibtex
@software{jameschapman19_2020_4382740,  
  author       = {jameschapman19},  
  title        = {jameschapman19/cca\_zoo: First pre-release},  
  month        = dec,  
  year         = 2020,  
  publisher    = {Zenodo},  
  version      = {v1.1.6},  
  doi          = {10.5281/zenodo.4382740},  
  url          = {https://doi.org/10.5281/zenodo.4382740}  
}
```

# Issues/Feedback
I've translated my work building baselines for my own research into a pytohn package for the experience of doing so. 
With that in mind if you have either suggestions for improvements/additions do let me know using the issues tab.
The intention is to give flexibility to build new algorithms and substitute model architectures but there is a tradeoff between robustness and flexibility.

# Sources
I've added this section to give due credit to the repositories that helped me in addition to their copyright notices in the code where relevant.

Models can be tested on data from MNIST datasets provided by the torch package (https://pytorch.org/) and the UCI dataset provided by mvlearn package (https://mvlearn.github.io/)

Other Implementations of DCCA/DCCAE:

Keras implementation of DCCA from @VahidooX's github page(https://github.com/VahidooX)
The following are the other implementations of DCCA in MATLAB and C++. These codes are written by the authors of the original paper:

Torch implementation of DCCA from @MichaelVll & @Arminarj: https://github.com/Michaelvll/DeepCCA

C++ implementation of DCCA from Galen Andrew's website (https://homes.cs.washington.edu/~galen/)

MATLAB implementation of DCCA/DCCAE from Weiran Wang's website (http://ttic.uchicago.edu/~wwang5/dccae.html)

Implementation of VAE:

Torch implementation of VAE (https://github.com/pytorch/examples/tree/master/vae)

Implementation of Sparse PLS:

MATLAB implementation of SPLS by @jmmonteiro (https://github.com/jmmonteiro/spls)
