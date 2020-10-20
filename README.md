Canonical Correlation Analysis: Linear, Kernel, Deep and Generative
This is a compilation of methods for CCA including linear (ALS/scikit-learn), rCCA (ridge penalty), sparseCCA (Witten/Parkhomenko), elasticCCA(waaijenborg), kernel methods (pyrcca), Deep CCA (Andrew), Deep Canonically Correlated Autoencoders(), Deep Variational CCA () and Deep Variational CCA_private ().

All models are evaluated on a noisy version of MNIST dataset with thanks to @VahidooX, @MichaelVll @Arminarj. The train/validation/test split is the original split of MNIST.

You can download them from noisymnist_view1.gz and noisymnist_view2.gz

The DCCAE is just a DCCA with reconstruction networks from the latent space and additional reconstruction losses. As such I implement both DCCA and DCCAE in a single model with a parameter weighting the reconstruction loss (i.e. 0 in the DCCA case)

The DVCCA (in layman's terms!) can be understood as an encoder-decoder with 3 loss terms: one that ensures a good encoding (this is the K-L divergence between the encoding and a N(0,1) gaussian), and two ensuring good reconstructions from the latent variable to the original views.

If you are familiar with the variational autoencoder, you will notice that 2 out of 3 of these losses are the same as the VAE set up with an additional reconstruction terms for the second view.

The DVCCA_private extends the DVCCA by adding private latent variables. This gives us 2 additional loss terms ensuring a 'good' encoding of the private information from each view (the KL divergence between the encoding and a N(0,1) gaussian). 

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