"""
# CCA_methods: Examples
In this notebook I demonstrate the general pipeline I use in the CCA_methods package.
"""

### Imports

import numpy as np
import CCA_methods
import itertools
import os
import matplotlib

matplotlib.use('TKAgg', warn=False, force=True)
import matplotlib.pyplot as plt

### Load MNIST Data

os.chdir('..')
train_set_1, val_set_1, test_set_1 = CCA_methods.mnist_utils.load_data('Data/noisymnist_view1.gz')
train_set_2, val_set_2, test_set_2 = CCA_methods.mnist_utils.load_data('Data/noisymnist_view2.gz')

train_set_1 = train_set_1[0][:1000]
train_set_2 = train_set_2[0][:1000]
val_set_1 = val_set_1[0][:1000]
val_set_2 = val_set_2[0][:1000]
test_set_1 = test_set_1[0][:1000]
test_set_2 = test_set_2[0][:1000]

print(train_set_1.shape)
print(train_set_2.shape)

### Settings

# The number of latent dimensions across models
latent_dims = 1
# The number of folds used for cross-validation/hyperparameter tuning
cv_folds = 5
# The number of iterations used for alternating least squares/iterative methods
max_als_iter = 5
# The number of epochs used for deep learning based models
epoch_num = 50

"""
### Linear CCA
We can do this via a few different methods
- alternating least squares
- generalized cca (equivalent to SVD/Eigendecomposition)
- multiset cca (equivalent to SVD/Eigendecomposition)
- scikit learn (NIPALS)

(Note that although the MNIST data here is not full rank,
both alternating least squares and NIPALS find least squares solutions
and therefore this problem is avoided)
"""
# %%
linear_cca = CCA_methods.linear.Wrapper(latent_dims=latent_dims)

linear_cca.fit(train_set_1, train_set_2)

linear_cca_results = np.stack(
    (linear_cca.train_correlations[0, 1], linear_cca.predict_corr(test_set_1, test_set_2)[0, 1]))

scikit_cca = CCA_methods.linear.Wrapper(latent_dims=latent_dims, method='scikit')

scikit_cca.fit(train_set_1, train_set_2)

scikit_cca_results = np.stack(
    (scikit_cca.train_correlations[0, 1], scikit_cca.predict_corr(test_set_1, test_set_2)[0, 1]))

gcca = CCA_methods.linear.Wrapper(latent_dims=latent_dims, method='gcca')

# small ammount of regularisation added since data is not full rank
params = {'c': [1, 1]}

gcca.fit(train_set_1, train_set_2, params=params)

gcca_results = np.stack((scikit_cca.train_correlations[0, 1], scikit_cca.predict_corr(test_set_1, test_set_2)[0, 1]))

"""
### Regularized CCA with hyperparameter tuning
- penalized matrix decomposition ('pmd')
- sparse cca/alternating lasso regression ('scca')
- ridge cca/alternating ridge regression ('l2')
- parkhomenko sparse cca ('parkhomenko')
- elastic ('elastic')

parameter candidates for cross validation are given as a list of lists as shown in the examples
"""
# %%
# PMD
c1 = [1, 3, 7, 9]
c2 = [1, 3, 7, 9]
param_candidates = {'c': list(itertools.product(c1, c2))}

pmd = CCA_methods.linear.Wrapper(latent_dims=latent_dims, method='pmd',
                                 max_iter=max_als_iter).cv_fit(train_set_1, train_set_2,
                                                               param_candidates=param_candidates,
                                                               folds=cv_folds, verbose=True)

pmd_results = np.stack((pmd.train_correlations[0, 1, :], pmd.predict_corr(test_set_1, test_set_2)[0, 1, :]))

# Elastic
c1 = [0.01, 0.1, 1]
c2 = [0.01, 0.1, 1]
l1_1 = [0.01, 0.01, 0.1]
l1_2 = [0.01, 0.01, 0.1]
param_candidates = {'c': list(itertools.product(c1, c2)), 'ratio': list(itertools.product(l1_1, l1_2))}

elastic = CCA_methods.linear.Wrapper(latent_dims=latent_dims, method='elastic',
                                     max_iter=max_als_iter).cv_fit(train_set_1, train_set_2,
                                                                   param_candidates=param_candidates,
                                                                   folds=cv_folds, verbose=True)

elastic_results = np.stack((elastic.train_correlations[0, 1, :], elastic.predict_corr(test_set_1, test_set_2)[0, 1, :]))

"""
### Kernel CCA

Similarly, we can use kernel CCA methods:
- regularized kernel CCA ('linear')
- sparse cca/alternating lasso regression ('poly')
- ridge cca/alternating ridge regression ('gaussian')
"""
# %%
# r-kernel cca
param_candidates = {'kernel': ['linear'], 'reg': [1e+4, 1e+5, 1e+6]}
kernel_reg = CCA_methods.linear.Wrapper(latent_dims=latent_dims, method='kernel',
                                        max_iter=max_als_iter).cv_fit(train_set_1, train_set_2,
                                                                      folds=cv_folds,
                                                                      param_candidates=param_candidates,
                                                                      verbose=True)
kernel_reg_results = np.stack((
    kernel_reg.train_correlations[0, 1, :],
    kernel_reg.predict_corr(test_set_1, test_set_2)[0, 1, :]))

# kernel cca (poly)
param_candidates = {'kernel': ['poly'], 'degree': [2, 3, 4], 'reg': [1e+6, 1e+7, 1e+8]}

kernel_poly = CCA_methods.linear.Wrapper(latent_dims=latent_dims, method='kernel',
                                         max_iter=max_als_iter).cv_fit(train_set_1, train_set_2,
                                                                       folds=cv_folds,
                                                                       param_candidates=param_candidates,
                                                                       verbose=True)

kernel_poly_results = np.stack((
    kernel_poly.train_correlations[0, 1, :],
    kernel_poly.predict_corr(test_set_1, test_set_2)[0, 1, :]))

# kernel cca (gaussian)
param_candidates = {'kernel': ['gaussian'], 'sigma': [1e+2, 1e+3], 'reg': [1e+6, 1e+7, 1e+8]}

kernel_gaussian = CCA_methods.linear.Wrapper(latent_dims=latent_dims, method='kernel',
                                             max_iter=max_als_iter).cv_fit(train_set_1, train_set_2,
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
- Deep Generalized CCA (DGCCA)

Both of the CCA loss and the GCCA loss can be used for DCCA/DCCAE since they are equivalent for two views.

To implement DCCA use DCCAE class with lam=0 (default). This multiplies the reconstruction loss term by 0.
"""
# %%
dcca = CCA_methods.deep.Wrapper(latent_dims=latent_dims, epoch_num=epoch_num, method='DCCAE',
                                loss_type='cca')

dcca.fit(train_set_1, train_set_2)

dcca_results = np.stack((dcca.train_correlations, dcca.predict_corr(test_set_1, test_set_2)))

dgcca = CCA_methods.deep.Wrapper(latent_dims=latent_dims, epoch_num=epoch_num, method='DCCAE',
                                 loss_type='gcca')

dgcca.fit(train_set_1, train_set_2)

dgcca_results = np.stack((dcca.train_correlations, dcca.predict_corr(test_set_1, test_set_2)))

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
dvcca = CCA_methods.deep.Wrapper(latent_dims=latent_dims, epoch_num=epoch_num, method='DVCCA', private=False)

dvcca.fit(train_set_1, train_set_2)

dvcca_results = np.stack((dvcca.train_correlations, dvcca.predict_corr(test_set_1, test_set_2)))

dvcca_p = CCA_methods.deep.Wrapper(latent_dims=latent_dims, epoch_num=epoch_num, method='DVCCA', private=True)

dvcca_p.fit(train_set_1, train_set_2)

dvcca_p_results = np.stack((dvcca_p.train_correlations, dvcca_p.predict_corr(test_set_1, test_set_2)))

"""
### Make results plot to compare methods
"""
# %%

all_labels = ['ALS', 'L2 - ALS', 'Witten', 'Parkhomenko', 'Waaijenborg - Elastic ALS',
              'scikit', 'DCCA']

all_results = np.stack(
    [linear_cca_results, pmd_results, elastic_results, kernel_reg_results, kernel_poly_results,
     kernel_gaussian_results, dcca_results, dgcca_results],
    axis=0)
all_labels = ['linear', 'pmd', 'elastic', 'linear kernel', 'polynomial kernel',
              'gaussian kernel', 'deep CCA', 'deep generalized CCA']

CCA_methods.plot_utils.plot_results(all_results, all_labels)
plt.show()
