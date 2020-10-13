import numpy as np
from sklearn.decomposition import PCA
import CCA_methods

def reduce_dimensions(train_x, train_y, test_x=None, test_y=None, components_x=100, components_y=100):
    pca = PCA(n_components=components_x)
    train_x = pca.fit_transform(train_x)
    if not test_x is None:
        test_x = pca.transform(test_x)

    pca = PCA(n_components=components_y)
    train_y = pca.fit_transform(train_y)
    if not test_y is None:
        test_y = pca.transform(test_y)
    return train_x, train_y, test_x, test_y

train_set_1, val_set_1, test_set_1 = CCA_methods.mnist_utils.load_data('Data/noisymnist_view1.gz')
train_set_2, val_set_2, test_set_2 = CCA_methods.mnist_utils.load_data('Data/noisymnist_view2.gz')

train_set_1 = train_set_1[0][:1000]
train_set_2 = train_set_2[0][:1000]
val_set_1 = val_set_1[0][:1000]
val_set_2 = val_set_2[0][:1000]
test_set_1 = test_set_1[0][:1000]
test_set_2 = test_set_2[0][:1000]

train_set_1, train_set_2, test_set_1, test_set_2 = reduce_dimensions(train_set_1, train_set_2,
                                                                             test_x=test_set_1,
                                                                             test_y=test_set_2)

outdim_size = 3
cv_folds = 1
max_als_iter = 100

# Deep FCN Torch
deep = CCA_methods.deep.Wrapper(outdim_size=outdim_size, epoch_num=1)

deep.fit(train_set_1, train_set_2)

deep_results = np.stack((deep.train_correlations, deep.predict_corr(test_set_1, test_set_2)))

all_labels = ['ALS', 'L2 - ALS', 'Witten', 'Parkhomenko', 'Waaijenborg - Elastic ALS',
              'scikit', 'DCCA']

# kernel cca
if len(train_set_1) < 1001:

    kernel_linear = CCA_methods.linear.Wrapper(outdim_size=outdim_size, method='kernel',
                                                       max_iter=max_als_iter, tol=1e-5).fit(train_set_1, train_set_2)

    kernel_linear_results = np.stack(
        (kernel_linear.train_correlations, kernel_linear.predict_corr(test_set_1, test_set_2)))

    # kernel cca (gaussian)
    param_candidates = {'kernel': ['gaussian'], 'gausigma': [1.0, 2.0, 3.0, 4.0], 'reg': [1e-4, 1e-3, 1e-2, 1e-1]}

    kernel_gaussian = CCA_methods.linear.Wrapper(outdim_size=outdim_size, method='kernel', max_iter=max_als_iter,
                                                         tol=1e-5).cv_fit(train_set_1, train_set_2, folds=cv_folds,
                                                                  param_candidates=param_candidates)

    kernel_gaussian_results = np.stack(
        (kernel_gaussian.train_correlations, kernel_gaussian.predict_corr(test_set_1, test_set_2)))

    # kernel cca (poly)
    param_candidates = {'kernel': ['poly'], 'degree': [2, 3, 4], 'reg': [1e-4, 1e-3, 1e-2, 1e-1]}

    kernel_poly = CCA_methods.linear.Wrapper(outdim_size=outdim_size, method='kernel', max_iter=max_als_iter,
                                                     tol=1e-5).cv_fit(train_set_1, train_set_2, folds=cv_folds,
                                                              param_candidates=param_candidates)

    kernel_poly_results = np.stack((kernel_poly.train_correlations, kernel_poly.predict_corr(test_set_1, test_set_2)))

# CCA_archive with ALS:
params = {'c_1': 0, 'c_2': 0}
basic_als = CCA_methods.linear.Wrapper(outdim_size=outdim_size, method='l2', params=params, max_iter=max_als_iter,
                                               tol=1e-5).fit(train_set_1, train_set_2)

basic_als_results = np.stack((basic_als.train_correlations, basic_als.predict_corr(test_set_1, test_set_2)))

# Regularized ALS
param_candidates = {'c_1': [0.1, 1, 10, 100], 'c_2': [0.1, 1, 10, 100]}

regularized_als = CCA_methods.linear.Wrapper(outdim_size=outdim_size, method='l2', max_iter=max_als_iter,
                                                     tol=1e-5).cv_fit(train_set_1, train_set_2,
                                                              param_candidates=param_candidates, folds=cv_folds)

regularized_als_results = np.stack(
    (regularized_als.train_correlations, regularized_als.predict_corr(test_set_1, test_set_2)))

# Sparse ALS (witten)
param_candidates = {'c_1': [1, 3, 7, 9], 'c_2': [1, 3, 7, 9]}

sparse_als_witten = CCA_methods.linear.Wrapper(outdim_size=outdim_size, method='witten',
                                                       max_iter=max_als_iter, tol=1e-5).cv_fit(train_set_1, train_set_2,
                                                                                       param_candidates=param_candidates,
                                                                                       folds=cv_folds)

sparse_als_witten_results = np.stack(
    (sparse_als_witten.train_correlations, sparse_als_witten.predict_corr(test_set_1, test_set_2)))

# Sparse ALS (parkohomenko)
param_candidates = {'c_1': [0.1, 0.2, 0.3, 0.4], 'c_2': [0.1, 0.2, 0.3, 0.4]}

sparse_als_parkohomenko = CCA_methods.linear.Wrapper(outdim_size=outdim_size, method='parkhomenko',
                                                             max_iter=max_als_iter, tol=1e-5).cv_fit(train_set_1, train_set_2,
                                                                                             param_candidates=param_candidates,
                                                                                             folds=cv_folds)

sparse_als_parkohomenko_results = np.stack(
    (sparse_als_parkohomenko.train_correlations, sparse_als_parkohomenko.predict_corr(test_set_1, test_set_2)))

# Sparse ALS (waaijenborg)
param_candidates = {'c_1': [1, 100], 'c_2': [1, 100], 'l1_ratio_1': [0.1, 0.5],
                    'l1_ratio_2': [0.1, 0.5]}

elastic_als = CCA_methods.linear.Wrapper(outdim_size=outdim_size, method='waaijenborg',
                                                 max_iter=max_als_iter, tol=1e-5).cv_fit(train_set_1, train_set_2,
                                                                                 param_candidates=param_candidates,
                                                                                 folds=cv_folds)

elastic_als_results = np.stack((elastic_als.train_correlations, elastic_als.predict_corr(test_set_1, test_set_2)))

## Scikit-learn
scikit = CCA_methods.linear.Wrapper(outdim_size=outdim_size, method='scikit', max_iter=max_als_iter, tol=1e-5).fit(
    train_set_1, train_set_2)

scikit_results = np.stack((scikit.train_correlations, scikit.predict_corr(test_set_1, test_set_2)))

all_results = np.stack(
    [basic_als_results, regularized_als_results, sparse_als_witten_results,
     sparse_als_parkohomenko_results, elastic_als_results,
     scikit_results, kernel_linear_results, kernel_gaussian_results, kernel_poly_results,
     deep_results],
    axis=0)
all_labels = ['ALS', 'L2 - ALS', 'Witten', 'Parkhomenko', 'Waaijenborg - Elastic ALS',
              'scikit', 'KCCA', 'KCCA-gaussian', 'KCCA-polynomial', 'DCCA']

CCA_methods.plot_utils.plot_results(all_results, all_labels)
