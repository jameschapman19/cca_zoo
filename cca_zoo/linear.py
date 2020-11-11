import itertools

import numpy as np
from hyperopt import tpe, fmin, Trials
from scipy.linalg import pinv2, block_diag, cholesky
from sklearn.cross_decomposition import CCA, PLSCanonical

import cca_zoo.KCCA
import cca_zoo.alternating_least_squares
import cca_zoo.generate_data
import cca_zoo.plot_utils


class Wrapper:
    """
    This is a wrapper class for linear, regularised and kernel  CCA, Multiset CCA and Generalized CCA.
    We create an instance with a method and number of latent dimensions.
    If we have more than 2 views we need to use generalized methods, but we can override in the 2 view case also with
    the generalized parameter.

    The class has a number of methods:

    fit(): gives us train correlations and stores the variables needed for out of sample prediction as well as some
    method-specific variables

    cv_fit(): allows us to perform a hyperparameter search and then fit the model using the optimal hyperparameters

    predict_corr(): allows us to predict the out of sample correlation for supplied views

    predict_view(): allows us to predict a reconstruction of missing views from the supplied views

    transform_view(): allows us to transform given views to the latent variable space

    remaining methods are used to
    """

    def __init__(self, latent_dims: int = 1, method: str = 'elastic', generalized: bool = False, max_iter: int = 200,
                 tol=1e-6):
        self.latent_dims = latent_dims
        self.method = method
        self.generalized = generalized
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, *args, params=None):
        self.params = {}
        if params is None:
            params = {}
        if len(args) > 2:
            self.generalized = True
            print('more than 2 views therefore switched to generalized')
        if 'c' not in params:
            c_dict = slicedict(params, 'c')
            if c_dict:
                self.params['c'] = list(c_dict.values())
            else:
                self.params['c'] = [0] * len(args)
        else:
            self.params['c'] = params['c']
        if 'l1_ratio' not in params:
            l1_dict = slicedict(params, 'l1_ratio')
            if l1_dict:
                self.params['l1_ratio'] = list(l1_dict.values())
            else:
                self.params['l1_ratio'] = [0] * len(args)
        else:
            self.params['l1_ratio'] = params['l1_ratio']
        if self.method == 'kernel':
            # Linear kernel by default
            self.params['kernel'] = params.get('kernel', 'linear')
            # First order polynomial by default
            self.params['degree'] = params.get('degree', 1)
            # First order polynomial by default
            self.params['sigma'] = params.get('sigma', 1.0)

        # Fit returns in-sample score vectors and correlations as well as models with transform functionality
        self.dataset_list = []
        self.dataset_means = []
        for dataset in args:
            self.dataset_means.append(dataset.mean(axis=0))
            self.dataset_list.append(dataset - dataset.mean(axis=0))

        if self.method == 'kernel':
            self.fit_kcca = cca_zoo.KCCA.KCCA(self.dataset_list[0], self.dataset_list[1], params=self.params,
                                              latent_dims=self.latent_dims)
            self.score_list = [self.fit_kcca.U, self.fit_kcca.V]
        elif self.method == 'pls':
            self.fit_scikit_pls(self.dataset_list[0], self.dataset_list[1])
        elif self.method == 'scikit':
            self.fit_scikit_cca(self.dataset_list[0], self.dataset_list[1])
        elif self.method == 'mcca':
            self.fit_mcca(*self.dataset_list)
        elif self.method == 'gcca':
            self.fit_gcca(*self.dataset_list)
        elif self.method == 'gep':
            self.fit_gep(*self.dataset_list)
        else:
            self.outer_loop(*self.dataset_list)
            if self.method[:4] == 'tree':
                self.tree_list = [self.tree_list[i] for i in range(len(args))]
                self.weights_list = [np.expand_dims(tree.feature_importances_, axis=1) for tree in self.tree_list]
            else:
                self.rotation_list = []
                for i in range(len(args)):
                    self.rotation_list.append(
                        self.weights_list[i] @ pinv2(self.loading_list[i].T @ self.weights_list[i], check_finite=False))
        self.train_correlations = self.predict_corr(*args)
        return self

    def gridsearch_fit(self, *args, param_candidates=None, folds: int = 5, verbose: bool = False):
        best_params = grid_search(*args, max_iter=self.max_iter, latent_dims=self.latent_dims, method=self.method,
                                  param_candidates=param_candidates, folds=folds,
                                  verbose=verbose, tol=self.tol)
        self.fit(*args, params=best_params)
        return self

    def bayes_fit(self, *args, space=None, folds: int = 5, verbose=True):

        trials = Trials()

        best_params = fmin(
            fn=cross_validate(*args, method=self.method, latent_dims=self.latent_dims, folds=folds,
                              verbose=verbose, max_iter=self.max_iter, tol=self.tol).score,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials,
        )
        self.fit(*args, params=best_params)
        return self

    def predict_corr(self, *args):
        # Takes two datasets and predicts their out of sample correlation using trained model
        transformed_views = self.transform_view(*args)
        all_corrs = []
        for x, y in itertools.product(transformed_views, repeat=2):
            all_corrs.append(np.diag(np.corrcoef(x.T, y.T)[:self.latent_dims, self.latent_dims:]))
        all_corrs = np.array(all_corrs).reshape((len(args), len(args), self.latent_dims))
        return all_corrs

    def predict_view(self, *args):
        # Regress original given views onto target
        transformed_views = self.transform_view(*args)

        # Get the regression from the training data with available views
        predicted_target = np.mean([transformed_views[i] for i in range(len(args)) if args[i] is not None], axis=0)

        predicted_views = []
        for i, view in enumerate(args):
            if view is None:
                predicted_views.append(predicted_target @ pinv2(self.weights_list[i]))
            else:
                predicted_views.append(view)
        for i, predicted_view in enumerate(predicted_views):
            predicted_views[i] += self.dataset_means[i]
        return predicted_views

    def transform_view(self, *args):
        # Demeaning
        new_views = []
        for i, new_view in enumerate(args):
            if new_view is None:
                new_views.append(None)
            else:
                new_views.append(new_view - self.dataset_means[i])

        if self.method == 'kernel':
            transformed_views = list(self.fit_kcca.transform(new_views[0], new_views[1]))
        elif self.method == 'pls':
            transformed_views = list(self.PLS.transform(new_views[0], new_views[1]))
        elif self.method[:4] == 'tree':
            transformed_views = []
            for i, new_view in enumerate(new_views):
                if new_view is None:
                    transformed_views.append(None)
                else:
                    transformed_views.append(self.tree_list[i].predict(new_view))
        else:
            transformed_views = []
            for i, new_view in enumerate(new_views):
                if new_view is None:
                    transformed_views.append(None)
                else:
                    transformed_views.append(new_view @ self.rotation_list[i])
        # d x n x k
        return transformed_views

    def outer_loop(self, *args):
        # list of d: p x k
        self.weights_list = [np.zeros((args[i].shape[1], self.latent_dims)) for i in range(len(args))]
        # list of d: n x k
        self.score_list = [np.zeros((args[i].shape[0], self.latent_dims)) for i in range(len(args))]
        # list of d:
        self.loading_list = [np.zeros((args[i].shape[1], self.latent_dims)) for i in range(len(args))]

        if len(args) == 2:
            C_train = args[0].T @ args[1]
            C_train_res = C_train.copy()
        else:
            C_train_res = None

        residuals = list(args)
        # For each of the dimensions
        for k in range(self.latent_dims):
            self.inner_loop = cca_zoo.alternating_least_squares.ALS_inner_loop(*residuals, C=C_train_res,
                                                                               generalized=self.generalized,
                                                                               params=self.params,
                                                                               method=self.method,
                                                                               max_iter=self.max_iter)
            for i in range(len(args)):
                if self.method[:4] == 'tree':
                    self.tree_list = self.inner_loop.weights
                else:
                    self.weights_list[i][:, k] = self.inner_loop.weights[i]
                    self.score_list[i][:, k] = self.inner_loop.scores[i, :]
                    self.loading_list[i][:, k] = residuals[i].T @ self.score_list[i][:, k] / np.linalg.norm(
                        self.score_list[i][:, k])
                    residuals[i] -= np.outer(self.score_list[i][:, k] / np.linalg.norm(self.score_list[i][:, k]),
                                             self.loading_list[i][:, k])
        return self

    def fit_scikit_cca(self, train_set_1, train_set_2):
        self.cca = CCA(n_components=self.latent_dims, scale=False)
        self.cca.fit(train_set_1, train_set_2)
        self.score_list = [self.cca.x_scores_, self.cca.y_scores_]
        self.weights_list = [self.cca.x_weights_, self.cca.y_weights_]
        self.loading_list = [self.cca.x_loadings_, self.cca.y_loadings_]
        self.rotation_list = [self.cca.x_rotations_, self.cca.y_rotations_]
        return self

    def fit_scikit_pls(self, train_set_1, train_set_2):
        self.PLS = PLSCanonical(n_components=self.latent_dims, scale=False)
        self.PLS.fit(train_set_1, train_set_2)
        self.score_list = [self.PLS.x_scores_, self.PLS.y_scores_]
        self.weights_list = [self.PLS.x_weights_, self.PLS.y_weights_]
        return self

    def fit_mcca(self, *args):
        all_views = np.concatenate(args, axis=1)
        C = all_views.T @ all_views
        # Can regularise by adding to diagonal
        D = block_diag(*[(1 - self.params['c'][i]) * m.T @ m + self.params['c'][i] * np.eye(m.shape[1]) for i, m in
                         enumerate(args)])
        C -= block_diag(*[m.T @ m for i, m in
                          enumerate(args)]) - D
        R = cholesky(D, lower=False)
        whitened = np.linalg.inv(R.T) @ C @ np.linalg.inv(R)
        [eigvals, eigvecs] = np.linalg.eig(whitened)
        idx = np.argsort(eigvals, axis=0)[::-1]
        eigvecs = eigvecs[:, idx].real
        eigvals = eigvals[idx].real
        eigvecs = np.linalg.inv(R) @ eigvecs
        splits = np.cumsum([0] + [view.shape[1] for view in args])
        self.weights_list = [eigvecs[splits[i]:splits[i + 1], :self.latent_dims] for i in range(len(args))]
        self.rotation_list = self.weights_list
        self.score_list = [self.dataset_list[i] @ self.weights_list[i] for i in range(len(args))]

    def fit_gep(self, *args):
        all_views = np.concatenate(args, axis=1)
        C = all_views.T @ all_views
        # Can regularise by adding to diagonal
        D = block_diag(*[(1 - self.params['c'][i]) * m.T @ m + self.params['c'][i] * np.eye(m.shape[1]) for i, m in
                         enumerate(args)])

        C -= block_diag(*[m.T @ m for i, m in
                          enumerate(args)])
        R = cholesky(D, lower=False)
        whitened = np.linalg.inv(R.T) @ C @ np.linalg.inv(R)
        [eigvals, eigvecs] = np.linalg.eig(whitened)
        idx = np.argsort(eigvals, axis=0)[::-1]
        eigvecs = eigvecs[:, idx].real
        eigvals = eigvals[idx].real
        eigvecs = np.linalg.inv(R) @ eigvecs
        splits = np.cumsum([0] + [view.shape[1] for view in args])
        self.weights_list = [eigvecs[splits[i]:splits[i + 1], :self.latent_dims] for i in range(len(args))]
        self.rotation_list = self.weights_list
        self.score_list = [self.dataset_list[i] @ self.weights_list[i] for i in range(len(args))]

    def fit_gcca(self, *args):
        Q = []
        for i, view in enumerate(args):
            view_cov = view.T @ view
            view_cov = (1 - self.params['c'][i]) * view_cov + self.params['c'][i] * np.eye(view_cov.shape[0])
            Q.append(view @ np.linalg.inv(view_cov) @ view.T)
        Q = np.sum(Q, axis=0)
        [eigvals, eigvecs] = np.linalg.eig(Q)
        idx = np.argsort(eigvals, axis=0)[::-1]
        eigvecs = eigvecs[:, idx].real
        eigvals = eigvals[idx].real
        self.weights_list = [np.linalg.pinv(view) @ eigvecs[:, :self.latent_dims] for view in args]
        self.rotation_list = self.weights_list
        self.score_list = [self.dataset_list[i] @ self.weights_list[i] for i in range(len(args))]


def permutation_test(train_set_1, train_set_2, latent_dims=5,
                     method='als', params=None, n_reps=100, level=0.05):
    if params is None:
        params = {}
    rho_train = np.zeros((n_reps, latent_dims))

    for _ in range(n_reps):
        print('permutation test rep: ', _ / n_reps, flush=True)
        results = Wrapper(latent_dims=latent_dims, method=method).fit(train_set_1, train_set_2,
                                                                      params=params).train_correlations
        np.random.shuffle(train_set_1)
        rho_train[_, :] = results

    p_vals = np.zeros(latent_dims)
    # FWE Adjusted
    for i in range(latent_dims):
        p_vals[i] = (1 + (rho_train[:, 0] > rho_train[0, i]).sum()) / n_reps
    hypothesis_test = False
    significant_dims = 0
    while not hypothesis_test:
        if p_vals[significant_dims] > level:
            hypothesis_test = True
        else:
            significant_dims += 1
        if significant_dims == len(p_vals):
            hypothesis_test = True

    print('significant dims at level: ', str(level * 100), '%:', str(significant_dims), flush=True)
    print(p_vals, flush=True)
    return p_vals, significant_dims


def slicedict(d, s):
    return {k: v for k, v in d.items() if k.startswith(s)}


def grid_search(*args, max_iter: int = 100, latent_dims: int = 5, method: str = 'l2', param_candidates=None,
                folds: int = 5,
                verbose=False, tol=1e-6):
    print('cross validation with ', method, flush=True)
    print('number of folds: ', folds, flush=True)

    # Set up an array for each set of hyperparameters
    assert (len(param_candidates) > 0)
    hyperparameter_grid_shape = [len(v) for k, v in param_candidates.items()]
    hyperparameter_scores = np.zeros(hyperparameter_grid_shape)

    for index, x in np.ndenumerate(hyperparameter_scores):
        params = {}
        p_num = 0
        for key in param_candidates.keys():
            params[key] = param_candidates[key][index[p_num]]
            p_num += 1
        hyperparameter_scores[index] = -cross_validate(*args, method=method, latent_dims=latent_dims, folds=folds,
                                                       verbose=verbose, max_iter=max_iter, tol=tol).score(params)

    # Find index of maximum value from 2D numpy array
    result = np.where(hyperparameter_scores == np.amax(hyperparameter_scores))
    # Return the 1st
    best_params = {}
    p_num = 0
    for key in param_candidates.keys():
        best_params[key] = param_candidates[key][result[p_num][0].item()]
        p_num += 1
    print('Best score : ', np.amax(hyperparameter_scores), flush=True)
    print(best_params, flush=True)
    if method == 'kernel':
        kernel_type = param_candidates.pop('kernel')[0]
        cca_zoo.plot_utils.cv_plot(hyperparameter_scores[0], param_candidates, method + ":" + kernel_type)
    elif not method == 'elastic':
        cca_zoo.plot_utils.cv_plot(hyperparameter_scores, param_candidates, method)
    return best_params


class cross_validate:
    def __init__(self, *args, latent_dims: int = 1, method: str = 'l2', generalized: bool = False, folds=5,
                 verbose=False, max_iter: int = 500,
                 tol=1e-6):
        self.latent_dims = latent_dims
        self.method = method
        self.generalized = generalized
        self.folds = folds
        self.verbose = verbose
        self.data = args
        self.max_iter = max_iter
        self.tol = tol

    def score(self, params):

        scores = np.zeros(self.folds)
        inds = np.arange(self.data[0].shape[0])
        np.random.shuffle(inds)
        if self.folds == 1:
            # If 1 fold do an 80:20 split
            fold_inds = np.array_split(inds, 5)
        else:
            fold_inds = np.array_split(inds, self.folds)
        for fold in range(self.folds):
            train_sets = [np.delete(data, fold_inds[fold], axis=0) for data in self.data]
            val_sets = [data[fold_inds[fold], :] for data in self.data]
            scores[fold] = \
                Wrapper(latent_dims=self.latent_dims, method=self.method, max_iter=self.max_iter, tol=self.tol).fit(
                    *train_sets, params=params).predict_corr(
                    *val_sets).sum(axis=-1)[np.triu_indices(len(self.data), 1)].sum()
        metric = scores.sum(axis=0) / self.folds
        if np.isnan(metric):
            metric = 0
        if self.verbose:
            print(params)
            print(metric)
        return -metric
