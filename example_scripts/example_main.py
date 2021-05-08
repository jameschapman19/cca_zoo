"""
# cca_zoo: Examples
In this script I demonstrate the general pipeline I use in the cca_zoo package.
"""

import itertools
import os

import matplotlib.pyplot as plt
# Imports
import numpy as np
from torch.utils.data import Subset

from cca_zoo.data import Noisy_MNIST_Dataset
from cca_zoo.models.wrappers import CCA, SCCA, ElasticCCA, PMD, PLS, KCCA, MCCA, GCCA, TCCA

# Load MNIST Data
os.chdir('..')
N = 1000
dataset = Noisy_MNIST_Dataset(mnist_type='MNIST', train=True)
ids = np.arange(min(2 * N, len(dataset)))
np.random.shuffle(ids)
train_ids, val_ids = np.array_split(ids, 2)
val_dataset = Subset(dataset, val_ids)
train_dataset = Subset(dataset, train_ids)
test_dataset = Noisy_MNIST_Dataset(mnist_type='MNIST', train=False)
test_ids = np.arange(min(N, len(test_dataset)))
np.random.shuffle(test_ids)
test_dataset = Subset(test_dataset, test_ids)
train_view_1, train_view_2, train_rotations, train_OH_labels, train_labels = train_dataset.dataset.to_numpy(
    train_dataset.indices)
val_view_1, val_view_2, val_rotations, val_OH_labels, val_labels = val_dataset.dataset.to_numpy(val_dataset.indices)
test_view_1, test_view_2, test_rotations, test_OH_labels, test_labels = test_dataset.dataset.to_numpy(
    test_dataset.indices)

# Settings

# The number of latent dimensions across models
latent_dims = 2
# The number of folds used for cross-validation/hyperparameter tuning
cv_folds = 3
# For running hyperparameter tuning in parallel (0 if not)
jobs = 2
# Number of iterations for iterative algorithms
max_iter = 10
# number of epochs for deep models
epochs = 50

# PMD
c1 = [1, 3, 7, 9]
c2 = [1, 3, 7, 9]
param_candidates = {'c': list(itertools.product(c1, c2))}

pmd = PMD(latent_dims=latent_dims, max_iter=max_iter).gridsearch_fit(
    train_view_1,
    train_view_2,
    param_candidates=param_candidates,
    folds=cv_folds,
    verbose=True, jobs=jobs,
    plot=True)

pmd_results = np.stack((pmd.train_correlations[0, 1, :], pmd.predict_corr(test_view_1, test_view_2)[0, 1, :]))

"""
### Linear CCA via alternating least squares (can pass more than 2 views)
"""

# %%
linear_cca = CCA(latent_dims=latent_dims)

linear_cca.fit(train_view_1, train_view_2)

linear_cca_results = np.stack(
    (linear_cca.train_correlations[0, 1], linear_cca.predict_corr(test_view_1, test_view_2)[0, 1]))

"""
### (Regularized) Generalized CCA via alternating least squares (can pass more than 2 views)
"""

gcca = GCCA(latent_dims=latent_dims, c=[1, 1])

gcca.fit(train_view_1, train_view_2)

gcca_results = np.stack((gcca.train_correlations[0, 1], gcca.predict_corr(test_view_1, test_view_2)[0, 1]))

"""
### (Regularized) Multiset CCA via alternating least squares (can pass more than 2 views)
"""

mcca = MCCA(latent_dims=latent_dims, c=[0.5, 0.5])
# small ammount of regularisation added since data is not full rank

mcca.fit(train_view_1, train_view_2)

mcca_results = np.stack((mcca.train_correlations[0, 1], mcca.predict_corr(test_view_1, test_view_2)[0, 1]))

"""
### (Regularized) Tensor CCA via alternating least squares (can pass more than 2 views)
"""

tcca = TCCA(latent_dims=latent_dims, c=[0.99, 0.99])
# small ammount of regularisation added since data is not full rank

tcca.fit(train_view_1, train_view_2)

tcca_results = np.stack((tcca.train_correlations[0, 1], tcca.predict_corr(test_view_1, test_view_2)[0, 1]))

"""
### PLS with scikit-learn (only permits 2 views)
"""

# %%
pls = PLS(latent_dims=latent_dims)

pls.fit(train_view_1, train_view_2)

pls_results = np.stack(
    (pls.train_correlations[0, 1], pls.predict_corr(test_view_1, test_view_2)[0, 1]))

"""
### Sparse CCA (Penalized Matrix Decomposition) (can pass more than 2 views)
"""

# PMD
c1 = [1, 3, 7, 9]
c2 = [1, 3, 7, 9]
param_candidates = {'c': list(itertools.product(c1, c2))}

pmd = PMD(latent_dims=latent_dims, max_iter=max_iter).gridsearch_fit(
    train_view_1,
    train_view_2,
    param_candidates=param_candidates,
    folds=cv_folds,
    verbose=True, jobs=jobs,
    plot=True)

pmd_results = np.stack((pmd.train_correlations[0, 1, :], pmd.predict_corr(test_view_1, test_view_2)[0, 1, :]))

"""
### Sparse CCA (can pass more than 2 views)
"""

# Sparse CCA
c1 = [0.00001, 0.0001]
c2 = [0.00001, 0.0001]
param_candidates = {'c': list(itertools.product(c1, c2))}

scca = SCCA(latent_dims=latent_dims, max_iter=max_iter).gridsearch_fit(
    train_view_1,
    train_view_2,
    param_candidates=param_candidates,
    folds=cv_folds,
    verbose=True,
    jobs=jobs, plot=True)

scca_results = np.stack(
    (scca.train_correlations[0, 1, :], scca.predict_corr(test_view_1, test_view_2)[0, 1, :]))

"""
# Sparse CCA with ADMM
c1 = [0.00001, 0.0001]
c2 = [0.00001, 0.0001]
param_candidates = {'c': list(itertools.product(c1, c2))}

scca_admm = wrappers.SCCA_ADMM(latent_dims=latent_dims, tol=1e-5, max_iter=max_iter).gridsearch_fit(
    train_view_1,
    train_view_2,
    param_candidates=param_candidates,
    folds=cv_folds,
    verbose=True,
    jobs=jobs, plot=True)

scca_admm_results = np.stack(
    (scca_admm.train_correlations[0, 1, :], scca_admm.predict_corr(test_view_1, test_view_2)[0, 1, :]))
"""

"""
### Elastic CCA (can pass more than 2 views)
"""

# Elastic CCA
c1 = [0.001, 0.0001]
c2 = [0.001, 0.0001]
l1_1 = [0.01, 0.1]
l1_2 = [0.01, 0.1]
param_candidates = {'c': list(itertools.product(c1, c2)), 'l1_ratio': list(itertools.product(l1_1, l1_2))}

elastic = ElasticCCA(latent_dims=latent_dims,
                     max_iter=max_iter).gridsearch_fit(train_view_1,
                                                       train_view_2,
                                                       param_candidates=param_candidates,
                                                       folds=cv_folds,
                                                       verbose=True,
                                                       jobs=jobs,
                                                       plot=True)

elastic_results = np.stack(
    (elastic.train_correlations[0, 1, :], elastic.predict_corr(test_view_1, test_view_2)[0, 1, :]))

"""
### Kernel CCA

Similarly, we can use kernel CCA methods with [method='kernel']

We can use different kernels and their associated parameters in a similar manner to before
- regularized linear kernel CCA: parameters :  'kernel'='linear', 0<'c'<1
- polynomial kernel CCA: parameters : 'kernel'='poly', 'degree', 0<'c'<1
- gaussian rbf kernel CCA: parameters : 'kernel'='gaussian', 'sigma', 0<'c'<1
"""
# %%
# r-kernel cca
c1 = [0.9, 0.99]
c2 = [0.9, 0.99]

param_candidates = {'kernel': ['linear', 'linear'], 'c': list(itertools.product(c1, c2))}

kernel_reg = KCCA(latent_dims=latent_dims).gridsearch_fit(train_view_1, train_view_2,
                                                          folds=cv_folds,
                                                          param_candidates=param_candidates,
                                                          verbose=True, jobs=jobs,
                                                          plot=True)
kernel_reg_results = np.stack((
    kernel_reg.train_correlations[0, 1, :],
    kernel_reg.predict_corr(test_view_1, test_view_2)[0, 1, :]))

# kernel cca (poly)
degree1 = [2, 3]
degree2 = [2, 3]

param_candidates = {'kernel': ['poly', 'poly'], 'degree': list(itertools.product(degree1, degree2)),
                    'c': list(itertools.product(c1, c2))}

kernel_poly = KCCA(latent_dims=latent_dims).gridsearch_fit(train_view_1, train_view_2,
                                                           folds=cv_folds,
                                                           param_candidates=param_candidates,
                                                           verbose=True, jobs=jobs,
                                                           plot=True)

kernel_poly_results = np.stack((
    kernel_poly.train_correlations[0, 1, :],
    kernel_poly.predict_corr(test_view_1, test_view_2)[0, 1, :]))

# kernel cca (gaussian)
gamma1 = [1e+1, 1e+2, 1e+3]
gamma2 = [1e+1, 1e+2, 1e+3]

param_candidates = {'kernel': ['rbf', 'rbf'], 'sigma': list(itertools.product(gamma1, gamma2)),
                    'c': list(itertools.product(c1, c2))}

kernel_gaussian = KCCA(latent_dims=latent_dims).gridsearch_fit(train_view_1, train_view_2,
                                                               folds=cv_folds,
                                                               param_candidates=param_candidates,
                                                               verbose=True, jobs=jobs,
                                                               plot=True)

kernel_gaussian_results = np.stack((
    kernel_gaussian.train_correlations[0, 1, :],
    kernel_gaussian.predict_corr(test_view_1, test_view_2)[0, 1, :]))

"""
### Deep Learning

We also have deep CCA methods (and autoencoder variants)
- Deep CCA (DCCA)
- Deep Canonically Correlated Autoencoders (DCCAE)

We introduce a Config class from configuration.py. This contains a number of default settings for running DCCA.

"""
from cca_zoo.deepmodels import dcca, dccae, dvcca, objectives, architectures, deepwrapper

# %%
# DCCA
print('DCCA')
encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=784)
encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=784)
dcca_model = dcca.DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2])

# hidden_layer_sizes are shown explicitly but these are also the defaults
dcca_model = deepwrapper.DeepWrapper(dcca_model)

dcca_model.fit(train_view_1, train_view_2, epochs=epochs)

dcca_results = np.stack((dcca_model.train_correlations[0, 1], dcca_model.predict_corr(test_view_1, test_view_2)[0, 1]))

# DGCCA
print('DGCCA')
encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=784)
encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=784)
dgcca_model = dcca.DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2], objective=objectives.GCCA)

# hidden_layer_sizes are shown explicitly but these are also the defaults
dgcca_model = deepwrapper.DeepWrapper(dgcca_model)

dgcca_model.fit((train_view_1, train_view_2), epochs=epochs)

dgcca_results = np.stack(
    (dgcca_model.train_correlations[0, 1], dgcca_model.predict_corr((test_view_1, test_view_2))[0, 1]))

# DMCCA
print('DMCCA')
encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=784)
encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=784)
dmcca_model = dcca.DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2], objective=objectives.MCCA)

# hidden_layer_sizes are shown explicitly but these are also the defaults
dmcca_model = deepwrapper.DeepWrapper(dmcca_model)

dmcca_model.fit((train_view_1, train_view_2), epochs=epochs)

dmcca_results = np.stack(
    (dmcca_model.train_correlations[0, 1], dmcca_model.predict_corr((test_view_1, test_view_2))[0, 1]))

# DCCA_NOI
print('DCCA by non-linear orthogonal iterations')
encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=784)
encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=784)
dcca_noi_model = dcca.DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2], als=True)

# hidden_layer_sizes are shown explicitly but these are also the defaults
dcca_noi_model = deepwrapper.DeepWrapper(dcca_noi_model)

dcca_noi_model.fit((train_view_1, train_view_2), epochs=epochs)

dcca_noi_results = np.stack(
    (dcca_noi_model.train_correlations[0, 1], dcca_noi_model.predict_corr(test_view_1, test_view_2)[0, 1]))

# DCCAE
print('DCCAE')
encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=784)
encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=784)
decoder_1 = architectures.Decoder(latent_dims=latent_dims, feature_size=784)
decoder_2 = architectures.Decoder(latent_dims=latent_dims, feature_size=784)
dccae_model = dccae.DCCAE(latent_dims=latent_dims, encoders=[encoder_1, encoder_2], decoders=[decoder_1, decoder_2])

# hidden_layer_sizes are shown explicitly but these are also the defaults
dccae_model = deepwrapper.DeepWrapper(dccae_model)

dccae_model.fit((train_view_1, train_view_2), epochs=epochs)

dccae_results = np.stack(
    (dccae_model.train_correlations[0, 1], dccae_model.predict_corr((test_view_1, test_view_2))[0, 1]))

"""
### Deep Variational Learning
Finally we have Deep Variational CCA methods.
- Deep Variational CCA (DVCCA)
- Deep Variational CCA - private (DVVCA_p)

These are both implemented by the DVCCA class with private=True/False and both_encoders=True/False. If both_encoders,
the encoder to the shared information Q(z_shared|x) is modelled for both x_1 and x_2 whereas if both_encoders is false
it is modelled for x_1 as in the paper
"""

# %%
# DVCCA (technically bi-DVCCA)
print('DVCCA')
encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=784, variational=True)
encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=784, variational=True)
decoder_1 = architectures.Decoder(latent_dims=latent_dims, feature_size=784, norm_output=True)
decoder_2 = architectures.Decoder(latent_dims=latent_dims, feature_size=784, norm_output=True)
dvcca_model = dvcca.DVCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2], decoders=[decoder_1, decoder_2],
                          private=False)

# hidden_layer_sizes are shown explicitly but these are also the defaults
dvcca_model = deepwrapper.DeepWrapper(dvcca_model)

dvcca_model.fit((train_view_1, train_view_2), epochs=epochs)

dvcca_model_results = np.stack(
    (dvcca_model.train_correlations[0, 1], dvcca_model.predict_corr((test_view_1, test_view_2))[0, 1]))

# DVCCA_private (technically bi-DVCCA_private)
print('DVCCA_private')
encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=784, variational=True)
encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=784, variational=True)
private_encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=784, variational=True)
private_encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=784, variational=True)
decoder_1 = architectures.Decoder(latent_dims=latent_dims * 2, feature_size=784, norm_output=True)
decoder_2 = architectures.Decoder(latent_dims=latent_dims * 2, feature_size=784, norm_output=True)
dvccap_model = dvcca.DVCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2], decoders=[decoder_1, decoder_2],
                           private_encoders=[private_encoder_1, private_encoder_2], private=True)

# hidden_layer_sizes are shown explicitly but these are also the defaults
dvccap_model = deepwrapper.DeepWrapper(dvccap_model)

dvccap_model.fit((train_view_1, train_view_2), epochs=epochs)

dvccap_model_results = np.stack(
    (dvccap_model.train_correlations[0, 1], dvccap_model.predict_corr((test_view_1, test_view_2))[0, 1]))

"""
### Convolutional Deep Learning

We can vary the encoder architecture from the default fcn to encoder/decoder based on the brainnetcnn architecture or a simple cnn
"""

print('Convolutional DCCA')
encoder_1 = architectures.CNNEncoder(latent_dims=latent_dims, channels=[3, 3])
encoder_2 = architectures.CNNEncoder(latent_dims=latent_dims, channels=[3, 3])
dcca_conv_model = dcca.DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2])

dcca_conv_model = deepwrapper.DeepWrapper(dcca_conv_model)

# to change the models used change the cfg.encoder_models. We implement a CNN_Encoder and CNN_decoder as well
# as some based on brainnet architecture in cca_zoo.deep_models. Equally you could pass your own encoder/decoder models

dcca_conv_model.fit((train_view_1.reshape((-1, 1, 28, 28)), train_view_2.reshape((-1, 1, 28, 28))), epochs=epochs)

dcca_conv_results = np.stack(
    (dcca_conv_model.train_correlations[0, 1], dcca_conv_model.predict_corr((test_view_1.reshape((-1, 1, 28, 28)),
                                                                             test_view_2.reshape(
                                                                                 (-1, 1, 28, 28))))[0, 1]))

"""
### Make results plot to compare methods
"""
# %%

all_results = np.stack(
    [linear_cca_results, gcca_results, mcca_results, pls_results, pmd_results, elastic_results,
     scca_results, kernel_reg_results, kernel_poly_results,
     kernel_gaussian_results, dcca_results, dgcca_results, dmcca_results, dccae_results, dvcca_model_results,
     dcca_conv_results],
    axis=0)
all_labels = ['linear', 'gcca', 'mcca', 'pls', 'pmd', 'elastic', 'scca', 'linear kernel', 'polynomial kernel',
              'gaussian kernel', 'deep CCA', 'deep generalized CCA', 'deep multiset CCA', 'deep CCAE', 'deep VCCA',
              'deep convolutional cca']

from cca_zoo.utils import plot_utils

plot_utils.plot_results(all_results, all_labels)
plt.show()
