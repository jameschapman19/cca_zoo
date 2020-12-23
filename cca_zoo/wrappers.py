import itertools
import copy
import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import pinv2, block_diag, cholesky
from sklearn.cross_decomposition import CCA, PLSCanonical
from abc import ABCMeta, abstractmethod

import cca_zoo.alsinnerloop
import cca_zoo.data
import cca_zoo.kcca
import cca_zoo.plot_utils


class CCA_Base(metaclass=ABCMeta):
    """
    This is a base class for linear, regularised and kernel  CCA, Multiset CCA and Generalized CCA.
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

    """
    @abstractmethod
    def __init__(self, latent_dims: int = 1, tol=1e-3):
        self.train_correlations = None
        self.latent_dims = latent_dims
        self.tol = tol

    @abstractmethod
    def fit(self, *views, params=None):
        pass

    @abstractmethod
    def transform_view(self, *views):
        pass

    def predict_view(self, *views):
        """

        :param views: numpy arrays or None separated by comma. Each view needs to have the same number of features as its
         corresponding view in the training data.
        :return: list of numpy arrays with same dimensions as the inputs
        """
        pass

    def predict_corr(self, *views):
        """
        :param views: numpy arrays separated by comma. Each view needs to have the same number of features as its
         corresponding view in the training data
        :return: numpy array containing correlations between each pair of views for each dimension (#views*#views*#latent_dimensions)
        """
        # Takes two views and predicts their out of sample correlation using trained model
        transformed_views = self.transform_view(*views)
        all_corrs = []
        for x, y in itertools.product(transformed_views, repeat=2):
            all_corrs.append(np.diag(np.corrcoef(x.T, y.T)[:self.latent_dims, self.latent_dims:]))
        all_corrs = np.array(all_corrs).reshape((len(views), len(views), self.latent_dims))
        return all_corrs

    def demean_data(self, *views):
        views_demeaned = []
        self.view_means = []
        for view in views:
            self.view_means.append(view.mean(axis=0))
            views_demeaned.append(view - view.mean(axis=0))
        return views_demeaned

    def gridsearch_fit(self, *views, param_candidates=None, folds: int = 5, verbose: bool = False, jobs: int = 0,
                       plot: bool = False):
        """
        Fits the model using a user defined grid search. Returns parameters/objects that allow out of sample transformation or prediction
        Supports parallel model training with jobs>0
        :param views: numpy arrays separated by comma e.g. fit(view_1,view_2,view_3, params=params)
        :param param_candidates: 
        :param folds: number of folds used for cross validation
        :param verbose: whether to return scores for each set of parameters
        :return: fit model with best parameters
        """""
        if verbose:
            print('cross validation', flush=True)
            print('number of folds: ', folds, flush=True)

        # Set up an array for each set of hyperparameters
        assert (len(param_candidates) > 0)
        param_names = list(param_candidates.keys())
        param_values = list(param_candidates.values())
        param_combinations = list(itertools.product(*param_values))

        param_sets = []
        for param_set in param_combinations:
            param_dict = {}
            for i, param_name in enumerate(param_names):
                param_dict[param_name] = param_set[i]
            param_sets.append(param_dict)

        cv = CrossValidate(self, folds=folds, verbose=verbose)

        if jobs > 0:
            scores = Parallel(n_jobs=jobs)(delayed(cv.score)(*views, params=param_set) for param_set in param_sets)
        else:
            scores = [cv.score(*views, params=param_set) for param_set in param_sets]
        max_index = scores.index(max(scores))

        print('Best score : ', max(scores), flush=True)
        print(param_sets[max_index], flush=True)
        if plot:
            cca_zoo.plot_utils.cv_plot(scores, param_sets, self.__class__.__name__)

        self.fit(*views, params=param_sets[max_index])
        return self

    """
    def bayes_fit(self, *views, space=None, folds: int = 5, verbose=True):
        :param views: numpy arrays separated by comma e.g. fit(view_1,view_2,view_3, params=params)
        :param space:
        :param folds: number of folds used for cross validation
        :param verbose: whether to return scores for each set of parameters
        :return: fit model with best parameters
        trials = Trials()

        best_params = fmin(
            fn=CrossValidate(*views, method=self.method, latent_dims=self.latent_dims, folds=folds,
                             verbose=verbose, max_iter=self.max_iter, tol=self.tol).score,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials,
        )
        self.fit(*views, params=best_params)
        return self
    """


class KCCA(CCA_Base):
    def __init__(self, latent_dims: int = 1, tol=1e-3):
        super().__init__(latent_dims=latent_dims, tol=tol)

    def fit(self, *views, params=None):
        """
        :param views:
        :param params:
        :return:
        """
        # Linear kernel by default
        params['kernel'] = params.get('kernel', 'linear')
        # First order polynomial by default
        params['degree'] = params.get('degree', 1)
        # First order polynomial by default
        params['sigma'] = params.get('sigma', 1.0)
        views = self.demean_data(*views)
        self.fit_kcca = cca_zoo.kcca.KCCA(views[0], views[1], params=params,
                                          latent_dims=self.latent_dims)
        self.score_list = [self.fit_kcca.U, self.fit_kcca.V]
        self.train_correlations = self.predict_corr(*views)
        return self

    def transform_view(self, *views):
        transformed_views = list(self.fit_kcca.transform(views[0], views[1]))
        return transformed_views


class MCCA(CCA_Base):
    def __init__(self, latent_dims: int = 1, tol=1e-3):
        super().__init__(latent_dims=latent_dims, tol=tol)

    def fit(self, *views, params=None):
        """
        :param views: numpy arrays separated by comma. Each view needs to have the same number of features as its
         corresponding view in the training data
        :return:
        """
        if params is None:
            self.params = {'c': [0 for _ in views]}
        else:
            self.params = params
        views = self.demean_data(*views)
        all_views = np.concatenate(views, axis=1)
        C = all_views.T @ all_views
        # Can regularise by adding to diagonal
        D = block_diag(*[(1 - self.params['c'][i]) * m.T @ m + self.params['c'][i] * np.eye(m.shape[1]) for i, m in
                         enumerate(views)])
        C -= block_diag(*[m.T @ m for i, m in
                          enumerate(views)]) - D
        R = cholesky(D, lower=False)
        whitened = np.linalg.inv(R.T) @ C @ np.linalg.inv(R)
        [eigvals, eigvecs] = np.linalg.eig(whitened)
        idx = np.argsort(eigvals, axis=0)[::-1]
        eigvecs = eigvecs[:, idx].real
        eigvals = eigvals[idx].real
        eigvecs = np.linalg.inv(R) @ eigvecs
        splits = np.cumsum([0] + [view.shape[1] for view in views])
        self.weights_list = [eigvecs[splits[i]:splits[i + 1], :self.latent_dims] for i in range(len(views))]
        self.rotation_list = self.weights_list
        self.score_list = [views[i] @ self.weights_list[i] for i in range(len(views))]
        self.train_correlations = self.predict_corr(*views)
        return self

    def transform_view(self, *views):
        transformed_views = []
        for i, view in enumerate(views):
            transformed_views.append(view @ self.rotation_list[i])
        return transformed_views


class GCCA(CCA_Base):
    def __init__(self, latent_dims: int = 1, tol=1e-3):
        super().__init__(latent_dims=latent_dims, tol=tol)

    def fit(self, *views, params=None):
        """
        :param views: numpy arrays separated by comma. Each view needs to have the same number of features as its
         corresponding view in the training data
        :return:
        """
        if params is None:
            self.params = {'c': [0 for _ in views]}
        else:
            self.params = params
        views = self.demean_data(*views)
        Q = []
        for i, view in enumerate(views):
            view_cov = view.T @ view
            view_cov = (1 - self.params['c'][i]) * view_cov + self.params['c'][i] * np.eye(view_cov.shape[0])
            Q.append(view @ np.linalg.inv(view_cov) @ view.T)
        Q = np.sum(Q, axis=0)
        [eigvals, eigvecs] = np.linalg.eig(Q)
        idx = np.argsort(eigvals, axis=0)[::-1]
        eigvecs = eigvecs[:, idx].real
        eigvals = eigvals[idx].real
        self.weights_list = [np.linalg.pinv(view) @ eigvecs[:, :self.latent_dims] for view in views]
        self.rotation_list = self.weights_list
        self.score_list = [views[i] @ self.weights_list[i] for i in range(len(views))]
        self.train_correlations = self.predict_corr(*views)
        return self

    def transform_view(self, *views):
        transformed_views = []
        for i, view in enumerate(views):
            transformed_views.append(view @ self.rotation_list[i])
        return transformed_views


class CCA_scikit(CCA_Base):
    def __init__(self, latent_dims: int = 1, tol=1e-3):
        super().__init__(latent_dims=latent_dims, tol=tol)

    def fit(self, *views, params=None):
        """
        :param train_set_1: numpy array
        :param train_set_2: numpy array with same number of samples as train_set_1
        :return:
        """
        views = self.demean_data(*views)
        self.cca = CCA(n_components=self.latent_dims, scale=False)
        self.cca.fit(*views)
        self.score_list = [self.cca.x_scores_, self.cca.y_scores_]
        self.weights_list = [self.cca.x_weights_, self.cca.y_weights_]
        self.loading_list = [self.cca.x_loadings_, self.cca.y_loadings_]
        self.rotation_list = [self.cca.x_rotations_, self.cca.y_rotations_]
        self.train_correlations = self.predict_corr(*views)
        return self

    def transform_view(self, *views):
        transformed_views = []
        for i, view in enumerate(views):
            transformed_views.append(view @ self.rotation_list[i])
        return transformed_views


class PLS_scikit(CCA_Base):
    def __init__(self, latent_dims: int = 1, tol=1e-3):
        super().__init__(latent_dims=latent_dims, tol=tol)

    def fit(self, *views, params=None):
        """
        :param train_set_1:
        :param train_set_2:
        :return:
        """
        views = self.demean_data(*views)
        self.PLS = PLSCanonical(n_components=self.latent_dims, scale=False)
        self.PLS.fit(*views)
        self.score_list = [self.PLS.x_scores_, self.PLS.y_scores_]
        self.weights_list = [self.PLS.x_weights_, self.PLS.y_weights_]
        self.train_correlations = self.predict_corr(*views)
        return self

    def transform_view(self, *views):
        transformed_views = list(self.PLS.transform(views[0], views[1]))
        return transformed_views


class CCA_ALS(CCA_Base):
    def __init__(self, latent_dims: int = 1, tol=1e-3, method='elastic', max_iter=100, generalized=False):
        super().__init__(latent_dims=latent_dims, tol=tol)
        self.method = method
        self.max_iter = max_iter
        self.generalized = generalized

    def fit(self, *views, params=None):
        """
        Fits the model for a given set of parameters (or use default values). Returns parameters/objects that allow out of sample transformation or prediction
        :param views: numpy arrays separated by comma e.g. fit(view_1,view_2,view_3, params=params)
        :param params: a dictionary containing the relevant parameters required for the model. If None use defaults
        :return: training data correlations and the parameters required to call other functions in the class.
        """
        views = self.demean_data(*views)
        self.params = {}
        if params is None:
            params = {}
        if len(views) > 2:
            self.generalized = True
            print('more than 2 views therefore switched to generalized')
        if 'c' not in params:
            c_dict = slicedict(params, 'c')
            if c_dict:
                self.params['c'] = list(c_dict.values())
            else:
                self.params['c'] = [0] * len(views)
        else:
            self.params['c'] = params['c']
        if 'l1_ratio' not in params:
            l1_dict = slicedict(params, 'l1_ratio')
            if l1_dict:
                self.params['l1_ratio'] = list(l1_dict.values())
            else:
                self.params['l1_ratio'] = [0] * len(views)
        else:
            self.params['l1_ratio'] = params['l1_ratio']

        self.outer_loop(*views)
        if self.method[:4] == 'tree':
            pass
            """
            self.tree_list = [self.tree_list[i] for i in range(len(views))]
            self.weights_list = [np.expand_dims(tree.feature_importances_, axis=1) for tree in self.tree_list]
            """
        else:
            self.rotation_list = []
            for i in range(len(views)):
                self.rotation_list.append(
                    self.weights_list[i] @ pinv2(self.loading_list[i].T @ self.weights_list[i], check_finite=False))
        self.train_correlations = self.predict_corr(*views)
        return self

    def transform_view(self, *views):
        transformed_views = []
        for i, view in enumerate(views):
            transformed_views.append(view @ self.rotation_list[i])
        return transformed_views

    def outer_loop(self, *views):
        """
        :param views: numpy arrays separated by comma. Each view needs to have the same number of features as its
         corresponding view in the training data
        :return: complete set of weights and scores required to calcuate train correlations of latent dimensions and
         transformations of out of sample data
        """
        # list of d: p x k
        self.weights_list = [np.zeros((views[i].shape[1], self.latent_dims)) for i in range(len(views))]
        # list of d: n x k
        self.score_list = [np.zeros((views[i].shape[0], self.latent_dims)) for i in range(len(views))]
        # list of d:
        self.loading_list = [np.zeros((views[i].shape[1], self.latent_dims)) for i in range(len(views))]

        residuals = copy.deepcopy(list(views))
        # For each of the dimensions
        for k in range(self.latent_dims):
            self.inner_loop = cca_zoo.alsinnerloop.AlsInnerLoop(*residuals,
                                                                generalized=self.generalized,
                                                                params=self.params,
                                                                method=self.method,
                                                                max_iter=self.max_iter)
            for i in range(len(residuals)):
                self.weights_list[i][:, k] = self.inner_loop.weights[i]
                self.score_list[i][:, k] = self.inner_loop.scores[i, :]
                self.loading_list[i][:, k] = residuals[i].T @ self.score_list[i][:, k] / np.linalg.norm(
                    self.score_list[i][:, k])
                residuals[i] -= np.outer(self.score_list[i][:, k] / np.linalg.norm(self.score_list[i][:, k]),
                                         self.loading_list[i][:, k])
        return self


def slicedict(d, s):
    """
    :param d: dictionary containing
    :param s:
    :return:
    """
    return {k: v for k, v in d.items() if k.startswith(s)}


class CrossValidate:
    def __init__(self, model: CCA_Base, folds: int = 5, verbose: bool = True):
        self.folds = folds
        self.verbose = verbose
        self.model = model

    def score(self, *views, params):
        """
        :param params:
        :return:
        """
        scores = np.zeros(self.folds)
        inds = np.arange(views[0].shape[0])
        np.random.shuffle(inds)
        if self.folds == 1:
            # If 1 fold do an 80:20 split
            fold_inds = np.array_split(inds, 5)
        else:
            fold_inds = np.array_split(inds, self.folds)
        for fold in range(self.folds):
            train_sets = [np.delete(view, fold_inds[fold], axis=0) for view in views]
            val_sets = [view[fold_inds[fold], :] for view in views]
            scores[fold] = self.model.fit(
                *train_sets, params=params).predict_corr(
                *val_sets).sum(axis=-1)[np.triu_indices(len(views), 1)].sum()
        metric = scores.sum(axis=0) / self.folds
        if np.isnan(metric):
            metric = 0
        if self.verbose:
            print(params)
            print(metric)
        return metric
