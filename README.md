# Canonical Correlation Analysis Methods: cca-zoo
## CCA, GCCA (Generalized CCA), MCCA (Multiset CCA), DCCA (Deep CCA), DGCCA (Deep Generalized CCA), DMCCA (Deep Multiset CCA), DVCCA (Deep Variational CCA), DCCAE (Deep Canonically Correlated Autoencoders), KCCA (Kernel CCA) and regularised variants using alternating least squares
### Documentation at https://cca-zoo.readthedocs.io/en/latest/
### Can be installed using pip install cca-zoo
### Recently added DCCA by non-linear orthogonal iterations (https://arxiv.org/pdf/1510.02054.pdf)

This is a compilation of methods for: 
-CCA (various implementations) 
-RCCA (ridge penalty)
-Sparse CCA (Witten/Parkhomenko,Waaijenborg)
-Kernel CCA (e.g. Hardoon)
-Deep CCA (Andrew) and DCCA by non-linear orthogonal iterations (Wang)
-Deep Canonically Correlated Autoencoders (Wang)
-Deep Variational CCA and Deep Variational CCA_private (Wang)

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
