"""
# cca_zoo: Examples
In this script I demonstrate the general pipeline I use in the cca_zoo package.
"""

### Imports

import numpy as np
import cca_zoo
import itertools
import os
from cca_zoo.configuration import Config
from sklearn import preprocessing
import matplotlib.pyplot as plt

### Load MNIST Data
os.chdir('..')

# if user has noisymnist_view1.gz in current directory
try:
    train_set_1, val_set_1, test_set_1 = cca_zoo.mnist_utils.load_data('noisymnist_view1.gz')
    train_set_2, val_set_2, test_set_2 = cca_zoo.mnist_utils.load_data('noisymnist_view2.gz')
    train_set_1 = train_set_1[0][:1000]
    train_set_2 = train_set_2[0][:1000]
    val_set_1 = val_set_1[0][:1000]
    val_set_2 = val_set_2[0][:1000]
    test_set_1 = test_set_1[0][:1000]
    test_set_2 = test_set_2[0][:1000]
except:
    # data_1, data_2, _, _ = cca_zoo.generate_data.generate_mai(3000, 1, 784, 784)
    data_1 = np.random.rand(3000, 700)
    data_2 = np.random.rand(3000, 700)
    train_set_1 = data_1[:1000]
    train_set_2 = data_2[:1000]
    val_set_1 = data_1[1000:2000]
    val_set_2 = data_2[1000:2000]
    test_set_1 = data_1[2000:3000]
    test_set_2 = data_2[2000:3000]

min_max_scaler = preprocessing.MinMaxScaler()
train_set_1 = min_max_scaler.fit_transform(train_set_1)
train_set_2 = min_max_scaler.fit_transform(train_set_2)
test_set_1 = min_max_scaler.fit_transform(test_set_1)
test_set_2 = min_max_scaler.fit_transform(test_set_2)

### Settings

# The number of latent dimensions across models
latent_dims = 2
# The number of folds used for cross-validation/hyperparameter tuning
cv_folds = 5

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

linear_cca.fit(train_set_1, train_set_2)

linear_cca_results = np.stack(
    (linear_cca.train_correlations[0, 1], linear_cca.predict_corr(test_set_1, test_set_2)[0, 1]))

scikit_cca = cca_zoo.wrapper.Wrapper(latent_dims=latent_dims, method='scikit')

scikit_cca.fit(train_set_1, train_set_2)

scikit_cca_results = np.stack(
    (scikit_cca.train_correlations[0, 1], scikit_cca.predict_corr(test_set_1, test_set_2)[0, 1]))

gcca = cca_zoo.wrapper.Wrapper(latent_dims=latent_dims, method='gcca')

# small ammount of regularisation added since data is not full rank
params = {'c': [1, 1]}

gcca.fit(train_set_1, train_set_2, params=params)

gcca_results = np.stack((scikit_cca.train_correlations[0, 1], scikit_cca.predict_corr(test_set_1, test_set_2)[0, 1]))

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
# %%
# PMD
c1 = [3, 7, 9]
c2 = [3, 7, 9]
param_candidates = {'c': list(itertools.product(c1, c2))}

pmd = cca_zoo.wrapper.Wrapper(latent_dims=latent_dims, method='pmd', tol=1e-5).gridsearch_fit(train_set_1, train_set_2,
                                                                                              param_candidates=param_candidates,
                                                                                              folds=cv_folds,
                                                                                              verbose=True)

pmd_results = np.stack((pmd.train_correlations[0, 1, :], pmd.predict_corr(test_set_1, test_set_2)[0, 1, :]))

# Elastic
c1 = [0.1, 1]
c2 = [0.1, 1]
l1_1 = [0.01, 0.1]
l1_2 = [0.01, 0.1]
param_candidates = {'c': list(itertools.product(c1, c2)), 'l1_ratio': list(itertools.product(l1_1, l1_2))}

elastic = cca_zoo.wrapper.Wrapper(latent_dims=latent_dims, method='elastic', tol=1e-5).gridsearch_fit(train_set_1,
                                                                                                      train_set_2,
                                                                                                      param_candidates=param_candidates,
                                                                                                      folds=cv_folds,
                                                                                                      verbose=True)

elastic_results = np.stack((elastic.train_correlations[0, 1, :], elastic.predict_corr(test_set_1, test_set_2)[0, 1, :]))

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
c1 = [0.5, 0.9]
c2 = [0.5, 0.9]

param_candidates = {'kernel': ['linear'], 'c': list(itertools.product(c1, c2))}

kernel_reg = cca_zoo.wrapper.Wrapper(latent_dims=latent_dims, method='kernel').gridsearch_fit(train_set_1, train_set_2,
                                                                                              folds=cv_folds,
                                                                                              param_candidates=param_candidates,
                                                                                              verbose=True)
kernel_reg_results = np.stack((
    kernel_reg.train_correlations[0, 1, :],
    kernel_reg.predict_corr(test_set_1, test_set_2)[0, 1, :]))

# kernel cca (poly)
param_candidates = {'kernel': ['poly'], 'degree': [2, 3], 'c': list(itertools.product(c1, c2))}

kernel_poly = cca_zoo.wrapper.Wrapper(latent_dims=latent_dims, method='kernel').gridsearch_fit(train_set_1, train_set_2,
                                                                                               folds=cv_folds,
                                                                                               param_candidates=param_candidates,
                                                                                               verbose=True)

kernel_poly_results = np.stack((
    kernel_poly.train_correlations[0, 1, :],
    kernel_poly.predict_corr(test_set_1, test_set_2)[0, 1, :]))

# kernel cca (gaussian)
param_candidates = {'kernel': ['rbf'], 'sigma': [1e+2, 1e+3], 'c': list(itertools.product(c1, c2))}

kernel_gaussian = cca_zoo.wrapper.Wrapper(latent_dims=latent_dims, method='kernel').gridsearch_fit(train_set_1,
                                                                                                   train_set_2,
                                                                                                   folds=cv_folds,
                                                                                                   param_candidates=param_candidates,
                                                                                                   verbose=True)

kernel_gaussian_results = np.stack((
    kernel_gaussian.train_correlations[0, 1, :],
    kernel_gaussian.predict_corr(test_set_1, test_set_2)[0, 1, :]))

"""
### Deep Learning

We also have deep CCA methods (and autoencoder variants)
- Deep CCA (DCCA)
- Deep Canonically Correlated Autoencoders (DCCAE)

We introduce a Config class from configuration.py. This contains a number of default settings for running DCCA.

"""
# %%
#DCCA
cfg = Config()
cfg.epoch_num = 100

# hidden_layer_sizes are shown explicitly but these are also the defaults
dcca = cca_zoo.deepwrapper.DeepWrapper(cfg)

dcca.fit(train_set_1, train_set_2)

dcca_results = np.stack((dcca.train_correlations, dcca.predict_corr(test_set_1, test_set_2)))

#DGCCA
#cfg.loss_type = cca_zoo.objectives.mcca
cfg.loss_type = cca_zoo.objectives.GCCA

# Note the different loss function
dgcca = cca_zoo.deepwrapper.DeepWrapper(cfg)

dgcca.fit(train_set_1, train_set_2)

dgcca_results = np.stack((dgcca.train_correlations, dgcca.predict_corr(test_set_1, test_set_2)))

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

dvcca.fit(train_set_1, train_set_2)

dvcca_results = np.stack((dvcca.train_correlations, dvcca.predict_corr(test_set_1, test_set_2)))


# DVCCA_private (technically bi-DVCCA_private)
#switch private=False default to private=True
cfg.private = True

dvcca_p = cca_zoo.deepwrapper.DeepWrapper(cfg)

dvcca_p.fit(train_set_1, train_set_2)

dvcca_p_results = np.stack((dvcca_p.train_correlations, dvcca_p.predict_corr(test_set_1, test_set_2)))

"""
### Convolutional Deep Learning

We can vary the encoder architecture from the default fcn to encoder/decoder based on the brainnetcnn architecture or a simple cnn
"""
# TODO Reshape the data to demonstrate CNN.

# %%
# cfg = Config()
# to change the models used change the cfg.encoder_models. We implement a CNN_Encoder and CNN_decoder as well
# as some based on brainnet architecture in cca_zoo.deep_models. Equally you could pass your own encoder/decoder models

# cfg.encoder_models = [cca_zoo.deep_models.CNN_Encoder, cca_zoo.deep_models.CNN_Encoder]
# cfg.hidden_layer_sizes = [[1, 1], [1, 1]]

# dcca_conv = cca_zoo.deep.Wrapper(cfg)

# dcca_conv.fit(train_set_1, train_set_2)

# dcca_conv_results = np.stack((dcca_conv.train_correlations, dcca_conv.predict_corr(test_set_1, test_set_2)))


"""
### Make results plot to compare methods
"""
# %%

all_results = np.stack(
    [linear_cca_results, pmd_results, elastic_results, kernel_reg_results, kernel_poly_results,
     kernel_gaussian_results, dcca_results, dgcca_results],
    axis=0)
all_labels = ['linear', 'pmd', 'elastic', 'linear kernel', 'polynomial kernel',
              'gaussian kernel', 'deep CCA', 'deep generalized CCA']

cca_zoo.plot_utils.plot_results(all_results, all_labels)
plt.show()
