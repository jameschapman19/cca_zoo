import itertools
import copy
import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import pinv2, block_diag, cholesky
from abc import abstractmethod
from sklearn.base import BaseEstimator
import cca_zoo.innerloop
import cca_zoo.data
import cca_zoo.kcca
import cca_zoo.plot_utils
from typing import Type
# from hyperopt import fmin, tpe, Trials

class CCA_Base(BaseEstimator):
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

    transform(): allows us to transform given views to the latent variable space

    """

    @abstractmethod
    def __init__(self, latent_dims: int = 1, tol=1e-5):
        self.train_correlations = None
        self.latent_dims = latent_dims
        self.tol = tol

    @abstractmethod
    def fit(self, *views, **kwargs):
        """
        The fit method takes any number of views as a numpy array along with associated parameters as a dictionary.
        Returns a fit model object which can be used to predict correlations or transform out of sample data.
        :param views: 2D numpy arrays for each view with the same number of rows (nxp)
        :param kwargs:
        :return:
        """
        pass
        return self

    @abstractmethod
    def transform(self, *views):
        """
        The fit method takes any number of views as a numpy array along with associated parameters as a dictionary.
        Returns a fit model object which can be used to predict correlations or transform out of sample data.
        :param views: 2D numpy arrays for each view with the same number of rows as well as the same number of columns as the data used in the most recent fit()
        :return:
        """
        pass
        return self

    def fit_transform(self, *views, **kwargs):
        """
        Apply fit and immediately transform the same data
        :param views:
        :param kwargs:
        :return:
        """
        self.fit(*views, **kwargs).transform(*views)

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
        transformed_views = self.transform(*views)
        all_corrs = []
        for x, y in itertools.product(transformed_views, repeat=2):
            all_corrs.append(np.diag(np.corrcoef(x.T, y.T)[:self.latent_dims, self.latent_dims:]))
        all_corrs = np.array(all_corrs).reshape((len(views), len(views), self.latent_dims))
        return all_corrs

    def demean_data(self, *views):
        """
        Since most methods require zero-mean data, demean_data() is used to demean training data as well as to apply this
        demeaning transformation to out of sample data
        :param views:
        :return:
        """
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
        :param views: numpy arrays separated by comma e.g. fit(view_1,view_2,view_3)
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
            scores = Parallel(n_jobs=jobs)(delayed(cv.score)(*views, **param_set) for param_set in param_sets)
        else:
            scores = [cv.score(*views, **param_set) for param_set in param_sets]
        max_index = scores.index(max(scores))

        print('Best score : ', max(scores), flush=True)
        print(param_sets[max_index], flush=True)
        if plot:
            cca_zoo.plot_utils.cv_plot(scores, param_sets, self.__class__.__name__)

        self.fit(*views, **param_sets[max_index])
        return self

    """
    def bayes_fit(self, *views, space=None, folds: int = 5, verbose=True):
        :param views: numpy arrays separated by comma e.g. fit(view_1,view_2,view_3)
        :param space:
        :param folds: number of folds used for cross validation
        :param verbose: whether to return scores for each set of parameters
        :return: fit model with best parameters
        trials = Trials()

        cv = CrossValidate(self, folds=folds, verbose=verbose)

        best_params = fmin(
            fn=cv.score(*views),
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials,
        )
        self.fit(*views, params=best_params)
        return self
    """


class KCCA(CCA_Base):
    def __init__(self, latent_dims: int = 1, tol=1e-5):
        super().__init__(latent_dims=latent_dims, tol=tol)

    def fit(self, *views, **kwargs):
        views = self.demean_data(*views)
        self.fit_kcca = cca_zoo.kcca.KCCA(*views, latent_dims=self.latent_dims, **kwargs)
        self.score_list = [self.fit_kcca.U, self.fit_kcca.V]
        self.train_correlations = self.predict_corr(*views)
        return self

    def transform(self, *views):
        transformed_views = list(self.fit_kcca.transform(views[0] - self.view_means[0], views[1] - self.view_means[1]))
        return transformed_views


class MCCA(CCA_Base):
    def __init__(self, latent_dims: int = 1, tol=1e-5):
        super().__init__(latent_dims=latent_dims, tol=tol)

    def fit(self, *views, **kwargs):
        """
        :param views: numpy arrays separated by comma. Each view needs to have the same number of features as its
         corresponding view in the training data
        :return:
        """
        self.c = kwargs.get('c', [0 for _ in views])
        assert (len(self.c) == len(views)), 'c requires as many values as #views'
        views = self.demean_data(*views)
        all_views = np.concatenate(views, axis=1)
        C = all_views.T @ all_views
        # Can regularise by adding to diagonal
        D = block_diag(*[(1 - self.c[i]) * m.T @ m + self.c[i] * np.eye(m.shape[1]) for i, m in
                         enumerate(views)])
        C -= block_diag(*[m.T @ m for i, m in
                          enumerate(views)]) - D
        R = cholesky(D, lower=False)
        whitened = np.linalg.inv(R.T) @ C @ np.linalg.inv(R)
        [eigvals, eigvecs] = np.linalg.eig(whitened)
        idx = np.argsort(eigvals, axis=0)[::-1]
        eigvecs = eigvecs[:, idx].real
        eigvecs = len(views) * np.linalg.inv(R) @ eigvecs
        splits = np.cumsum([0] + [view.shape[1] for view in views])
        self.weights_list = [eigvecs[split:splits[i + 1], :self.latent_dims] for i, split in enumerate(splits[:-1])]
        self.rotation_list = self.weights_list
        self.score_list = [view @ self.weights_list[i] for i, view in enumerate(views)]
        self.train_correlations = self.predict_corr(*views)
        return self

    def transform(self, *views):
        transformed_views = []
        for i, view in enumerate(views):
            transformed_views.append((view - self.view_means[0]) @ self.rotation_list[i])
        return transformed_views


class GCCA(CCA_Base):
    def __init__(self, latent_dims: int = 1, tol=1e-5):
        super().__init__(latent_dims=latent_dims, tol=tol)

    def fit(self, *views, **kwargs):
        """
        :param views: numpy arrays separated by comma. Each view needs to have the same number of features as its
         corresponding view in the training data
        :return:
        """
        self.c = kwargs.get('c', [0 for _ in views])
        assert (len(self.c) == len(views)), 'c requires as many values as #views'
        views = self.demean_data(*views)
        Q = []
        for i, view in enumerate(views):
            view_cov = view.T @ view
            view_cov = (1 - self.c[i]) * view_cov + self.c[i] * np.eye(view_cov.shape[0])
            Q.append(view @ np.linalg.inv(view_cov) @ view.T)
        Q = np.sum(Q, axis=0)
        [eigvals, eigvecs] = np.linalg.eig(Q)
        idx = np.argsort(eigvals, axis=0)[::-1]
        eigvecs = eigvecs[:, idx].real
        eigvals = eigvals[idx].real
        self.weights_list = [np.linalg.pinv(view) @ eigvecs[:, :self.latent_dims] for view in views]
        self.rotation_list = self.weights_list
        self.score_list = [view @ self.weights_list[i] for i, view in enumerate(views)]
        self.train_correlations = self.predict_corr(*views)
        return self

    def transform(self, *views):
        transformed_views = []
        for i, view in enumerate(views):
            transformed_views.append((view - self.view_means[i]) @ self.rotation_list[i])
        return transformed_views


class CCA_Iterative(CCA_Base):
    def __init__(self, inner_loop: Type[cca_zoo.innerloop.InnerLoop] = cca_zoo.innerloop.InnerLoop,
                 latent_dims: int = 1, tol=1e-5, max_iter=100):
        super().__init__(latent_dims=latent_dims, tol=tol)
        self.inner_loop = inner_loop
        self.max_iter = max_iter

    def fit(self, *views, **kwargs):
        """
        Fits the model for a given set of parameters (or use default values). Returns parameters/objects that allow out of sample transformation or prediction
        :param views: numpy arrays separated by comma e.g. fit(view_1,view_2,view_3)
        :param kwargs: a dictionary containing the relevant parameters required for the model. If None use defaults
        :return: training data correlations and the parameters required to call other functions in the class.
        """
        views = self.demean_data(*views)

        self.outer_loop(*views, **kwargs)
        self.rotation_list = []
        for i in range(len(views)):
            self.rotation_list.append(
                self.weights_list[i] @ pinv2(self.loading_list[i].T @ self.weights_list[i], check_finite=False))
        self.train_correlations = self.predict_corr(*views)
        return self

    def transform(self, *views):
        transformed_views = []
        for i, view in enumerate(views):
            transformed_views.append((view - self.view_means[i]) @ self.rotation_list[i])
        return transformed_views

    def outer_loop(self, *views, **kwargs):
        """
        :param views: numpy arrays separated by comma. Each view needs to have the same number of features as its
         corresponding view in the training data
        :return: complete set of weights and scores required to calcuate train correlations of latent dimensions and
         transformations of out of sample data
        """
        # list of d: p x k
        self.weights_list = [np.zeros((view.shape[1], self.latent_dims)) for view in views]
        # list of d: n x k
        self.score_list = [np.zeros((view.shape[0], self.latent_dims)) for view in views]
        # list of d:
        self.loading_list = [np.zeros((view.shape[1], self.latent_dims)) for view in views]

        residuals = copy.deepcopy(list(views))
        # For each of the dimensions
        for k in range(self.latent_dims):
            self.loop = self.inner_loop(*residuals, **kwargs)
            for i, residual in enumerate(residuals):
                self.weights_list[i][:, k] = self.loop.weights[i]
                self.score_list[i][:, k] = self.loop.scores[i, :]
                self.loading_list[i][:, k] = residual.T @ self.score_list[i][:, k] / np.linalg.norm(
                    self.score_list[i][:, k])
                residual -= np.outer(self.score_list[i][:, k] / np.linalg.norm(self.score_list[i][:, k]),
                                     self.loading_list[i][:, k])
        return self


class PLS(CCA_Iterative):
    def __init__(self, latent_dims: int = 1, tol=1e-5, max_iter=100, generalized=False):
        super().__init__(cca_zoo.innerloop.PLSInnerLoop, latent_dims, tol, max_iter)


class CCA(CCA_Iterative):
    def __init__(self, latent_dims: int = 1, tol=1e-5, max_iter=100, generalized=False):
        super().__init__(cca_zoo.innerloop.CCAInnerLoop, latent_dims, tol, max_iter)


class PMD(CCA_Iterative):
    def __init__(self, latent_dims: int = 1, tol=1e-5, max_iter=100, generalized=False):
        super().__init__(cca_zoo.innerloop.PMDInnerLoop, latent_dims, tol, max_iter)


class SCCA(CCA_Iterative):
    def __init__(self, latent_dims: int = 1, tol=1e-5, max_iter=100, generalized=False):
        super().__init__(cca_zoo.innerloop.PLSInnerLoop, latent_dims, tol, max_iter)


class SCCA_ADMM(CCA_Iterative):
    def __init__(self, latent_dims: int = 1, tol=1e-5, max_iter=100, generalized=False):
        super().__init__(cca_zoo.innerloop.ADMMInnerLoop, latent_dims, tol, max_iter)


class ElasticCCA(CCA_Iterative):
    def __init__(self, latent_dims: int = 1, tol=1e-5, max_iter=100, generalized=False):
        super().__init__(cca_zoo.innerloop.CCAInnerLoop, latent_dims, tol, max_iter)


def slicedict(d, s):
    """
    :param d: dictionary containing e.g. c_1, c_2
    :param s:
    :return:
    """
    return {k: v for k, v in d.items() if k.startswith(s)}


class CrossValidate:
    def __init__(self, model: CCA_Base, folds: int = 5, verbose: bool = True):
        self.folds = folds
        self.verbose = verbose
        self.model = model

    def score(self, *views, **kwargs):
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
                *train_sets, **kwargs).predict_corr(
                *val_sets).sum(axis=-1)[np.triu_indices(len(views), 1)].sum()
        metric = scores.sum(axis=0) / self.folds
        if np.isnan(metric):
            metric = 0
        if self.verbose:
            print(kwargs)
            print(metric)
        return metric
