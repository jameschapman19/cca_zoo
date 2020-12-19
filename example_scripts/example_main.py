"""
# cca_zoo: Examples
In this script I demonstrate the general pipeline I use in the cca_zoo package.
"""

# Imports
import numpy as np
import cca_zoo
import itertools
import os
from cca_zoo.configuration import Config
import matplotlib.pyplot as plt
from torch.utils.data import Subset

# Load MNIST Data
os.chdir('..')
N = 1000
dataset = cca_zoo.data.Noisy_MNIST_Dataset(mnist_type='FashionMNIST', train=True)
ids = np.arange(min(2 * N, len(dataset)))
np.random.shuffle(ids)
train_ids, val_ids = np.array_split(ids, 2)
val_dataset = Subset(dataset, val_ids)
train_dataset = Subset(dataset, train_ids)
test_dataset = cca_zoo.data.Noisy_MNIST_Dataset(mnist_type='FashionMNIST', train=False)
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
cv_folds = 5
# For running hyperparameter tuning in parallel (0 if not)
jobs = 1

"""
### Linear CCA
We can do this via a few different but equivalent methods when unregularized
- alternating least squares (default) [method='elastic']
- generalized cca (equivalent to SVD/Eigendecomposition) [method='gcca']
- multiset cca (equivalent to SVD/Eigendecomposition) [method='mcca']
- scikit learn (NIPALS) [method='scikit']
- generalized eigenvalue problem [method='gep']

(Note that although the MNIST data here is not full rank,
both alternating least squares and NIPALS find least squares solutions
and therefore this problem is avoided)
"""

# %%
linear_cca = cca_zoo.wrapper.Wrapper(latent_dims=latent_dims)

linear_cca.fit(train_view_1, train_view_2)

linear_cca_results = np.stack(
    (linear_cca.train_correlations[0, 1], linear_cca.predict_corr(test_view_1, test_view_2)[0, 1]))

scikit_cca = cca_zoo.wrapper.Wrapper(latent_dims=latent_dims, method='scikit')

scikit_cca.fit(train_view_1, train_view_2)

scikit_cca_results = np.stack(
    (scikit_cca.train_correlations[0, 1], scikit_cca.predict_corr(test_view_1, test_view_2)[0, 1]))

gcca = cca_zoo.wrapper.Wrapper(latent_dims=latent_dims, method='gcca')
# small ammount of regularisation added since data is not full rank
params = {'c': [1, 1]}

gcca.fit(train_view_1, train_view_2, params=params)

gcca_results = np.stack((gcca.train_correlations[0, 1], gcca.predict_corr(test_view_1, test_view_2)[0, 1]))

mcca = cca_zoo.wrapper.Wrapper(latent_dims=latent_dims, method='mcca')
# small ammount of regularisation added since data is not full rank
params = {'c': [0.5, 0.5]}

mcca.fit(train_view_1, train_view_2, params=params)

mcca_results = np.stack((mcca.train_correlations[0, 1], mcca.predict_corr(test_view_1, test_view_2)[0, 1]))

# %%
pls = cca_zoo.wrapper.Wrapper(latent_dims=latent_dims, method='pls')

pls.fit(train_view_1, train_view_2)

pls_results = np.stack(
    (pls.train_correlations[0, 1], pls.predict_corr(test_view_1, test_view_2)[0, 1]))

"""
### Regularized CCA with hyperparameter tuning
- penalized matrix decomposition [method='pmd'] : parameters: 1<'c'<sqrt(features)
- sparse cca/alternating lasso regression [method='scca'] : parameters: 'c'
- ridge cca/alternating ridge regression [method='elastic'] : parameters: 'c' 0<'l1_ratio'<1
- parkhomenko sparse cca [method='parkhomenko'] : parameters: 'c'

We can either fit a model with fixed parameters or use gridsearch_fit() to search over param_candidates,
where param_candidates is a dictionary of params with a list of lists for each view.

parameter candidates for cross validation are given as a list of lists as shown in the examples
"""

# PMD
c1 = [1, 3, 7, 9]
c2 = [1, 3, 7, 9]
param_candidates = {'c': list(itertools.product(c1, c2))}

pmd = cca_zoo.wrapper.Wrapper(latent_dims=latent_dims, method='pmd', tol=1e-5).gridsearch_fit(train_view_1,
                                                                                              train_view_2,
                                                                                              param_candidates=param_candidates,
                                                                                              folds=cv_folds,
                                                                                              verbose=True, jobs=jobs,
                                                                                              plot=True)

pmd_results = np.stack((pmd.train_correlations[0, 1, :], pmd.predict_corr(test_view_1, test_view_2)[0, 1, :]))

# Elastic CCA
c1 = [0.0001, 0.001]
c2 = [0.0001, 0.001]
l1_1 = [0.01, 0.1]
l1_2 = [0.01, 0.1]
param_candidates = {'c': list(itertools.product(c1, c2)), 'l1_ratio': list(itertools.product(l1_1, l1_2))}

elastic = cca_zoo.wrapper.Wrapper(latent_dims=latent_dims, method='elastic', tol=1e-5).gridsearch_fit(train_view_1,
                                                                                                      train_view_2,
                                                                                                      param_candidates=param_candidates,
                                                                                                      folds=cv_folds,
                                                                                                      verbose=True,
                                                                                                      jobs=jobs, plot=True)

elastic_results = np.stack(
    (elastic.train_correlations[0, 1, :], elastic.predict_corr(test_view_1, test_view_2)[0, 1, :]))

# Sparse CCA
c1 = [0.0001, 0.001]
c2 = [0.0001, 0.001]
param_candidates = {'c': list(itertools.product(c1, c2))}

scca = cca_zoo.wrapper.Wrapper(latent_dims=latent_dims, method='scca', tol=1e-5).gridsearch_fit(train_view_1,
                                                                                                train_view_2,
                                                                                                param_candidates=param_candidates,
                                                                                                folds=cv_folds,
                                                                                                verbose=True,
                                                                                                jobs=jobs, plot=True)

scca_results = np.stack(
    (scca.train_correlations[0, 1, :], scca.predict_corr(test_view_1, test_view_2)[0, 1, :]))

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

param_candidates = {'kernel': ['linear'], 'c': list(itertools.product(c1, c2))}

kernel_reg = cca_zoo.wrapper.Wrapper(latent_dims=latent_dims, method='kernel').gridsearch_fit(train_view_1,
                                                                                              train_view_2,
                                                                                              folds=cv_folds,
                                                                                              param_candidates=param_candidates,
                                                                                              verbose=True, jobs=jobs,
                                                                                              plot=True)
kernel_reg_results = np.stack((
    kernel_reg.train_correlations[0, 1, :],
    kernel_reg.predict_corr(test_view_1, test_view_2)[0, 1, :]))

# kernel cca (poly)
param_candidates = {'kernel': ['poly'], 'degree': [2, 3], 'c': list(itertools.product(c1, c2))}

kernel_poly = cca_zoo.wrapper.Wrapper(latent_dims=latent_dims, method='kernel').gridsearch_fit(train_view_1,
                                                                                               train_view_2,
                                                                                               folds=cv_folds,
                                                                                               param_candidates=param_candidates,
                                                                                               verbose=True, jobs=jobs,
                                                                                               plot=True)

kernel_poly_results = np.stack((
    kernel_poly.train_correlations[0, 1, :],
    kernel_poly.predict_corr(test_view_1, test_view_2)[0, 1, :]))

# kernel cca (gaussian)
param_candidates = {'kernel': ['rbf'], 'sigma': [1e+1, 1e+2, 1e+3], 'c': list(itertools.product(c1, c2))}

kernel_gaussian = cca_zoo.wrapper.Wrapper(latent_dims=latent_dims, method='kernel').gridsearch_fit(train_view_1,
                                                                                                   train_view_2,
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
# %%
# DCCA
cfg = Config()
cfg.epoch_num = 100

# hidden_layer_sizes are shown explicitly but these are also the defaults
dcca = cca_zoo.deepwrapper.DeepWrapper(cfg)

dcca.fit(train_view_1, train_view_2)

dcca_results = np.stack((dcca.train_correlations, dcca.predict_corr(test_view_1, test_view_2)))

# DGCCA
# cfg.loss_type = cca_zoo.objectives.mcca
cfg.loss_type = cca_zoo.objectives.GCCA

# Note the different loss function
dgcca = cca_zoo.deepwrapper.DeepWrapper(cfg)

dgcca.fit(train_view_1, train_view_2)

dgcca_results = np.stack((dgcca.train_correlations, dgcca.predict_corr(test_view_1, test_view_2)))

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
cfg = Config()
cfg.method = cca_zoo.dvcca.DVCCA
cfg.epoch_num = 100
dvcca = cca_zoo.deepwrapper.DeepWrapper(cfg)

dvcca.fit(train_view_1, train_view_2)

dvcca_results = np.stack((dvcca.train_correlations, dvcca.predict_corr(test_view_1, test_view_2)))

# DVCCA_private (technically bi-DVCCA_private)
# switch private=False default to private=True
cfg.private = True

dvcca_p = cca_zoo.deepwrapper.DeepWrapper(cfg)

dvcca_p.fit(train_view_1, train_view_2)

dvcca_p_results = np.stack((dvcca_p.train_correlations, dvcca_p.predict_corr(test_view_1, test_view_2)))

"""
### Convolutional Deep Learning

We can vary the encoder architecture from the default fcn to encoder/decoder based on the brainnetcnn architecture or a simple cnn
"""
# %%
cfg = Config()
cfg.epoch_num = 100
cfg.encoder_models = [cca_zoo.deep_models.CNNEncoder, cca_zoo.deep_models.CNNEncoder]
cfg.encoder_args = [{'channels':[3,3]},{'channels':[3,3]}]
# to change the models used change the cfg.encoder_models. We implement a CNN_Encoder and CNN_decoder as well
# as some based on brainnet architecture in cca_zoo.deep_models. Equally you could pass your own encoder/decoder models

dcca_conv = cca_zoo.deepwrapper.DeepWrapper(cfg)

dcca_conv.fit(train_view_1.reshape((-1, 1, 28, 28)), train_view_2.reshape((-1, 1, 28, 28)))

dcca_conv_results = np.stack((dcca_conv.train_correlations, dcca_conv.predict_corr(test_view_1.reshape((-1, 1, 28, 28)),
                                                                                   test_view_2.reshape(
                                                                                       (-1, 1, 28, 28)))))

"""
### Make results plot to compare methods
"""
# %%

all_results = np.stack(
    [linear_cca_results, scikit_cca_results, gcca_results,mcca_results, pls_results, scca_results, pmd_results, elastic_results,
     kernel_reg_results,
     kernel_poly_results,
     kernel_gaussian_results, dcca_results, dgcca_results, dcca_conv_results],
    axis=0)
all_labels = ['linear', 'scikit', 'gcca', 'mcca','pls', 'pmd', 'elastic', 'scca', 'linear kernel', 'polynomial kernel',
              'gaussian kernel', 'deep CCA', 'deep generalized CCA', 'deep convolutional cca']

cca_zoo.plot_utils.plot_results(all_results, all_labels)
plt.show()
