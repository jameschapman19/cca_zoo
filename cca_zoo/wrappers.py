"""CCA Models"""

import copy
import itertools
from abc import abstractmethod
from typing import Tuple

import numpy as np
import numpy.ma as ma
import pandas as pd
import tensorly as tl
from joblib import Parallel, delayed
from scipy.linalg import block_diag, eigh
from scipy.linalg import sqrtm
from sklearn.base import BaseEstimator
from tensorly.decomposition import parafac

import cca_zoo.data
import cca_zoo.innerloop
import cca_zoo.kcca
import cca_zoo.plot_utils


# from hyperopt import fmin, tpe, Trials

class _CCA_Base(BaseEstimator):
    """
    A class used to represent an Animal
    """

    @abstractmethod
    def __init__(self, latent_dims: int = 1):
        """

        Constructor for _CCA_Base
        :param latent_dims:
        """
        self.weights_list = None
        self.train_correlations = None
        self.latent_dims = latent_dims

    @abstractmethod
    def fit(self, *views: Tuple[np.ndarray, ...]):
        """
        Fits a given model

        :param views: numpy arrays with the same number of rows (samples) separated by commas
        """
        pass
        return self

    def transform(self, *views: Tuple[np.ndarray, ...], **kwargs):
        """
        Transforms data given a fit model

        :param views: numpy arrays with the same number of rows (samples) separated by commas
        :param kwargs: any additional keyword arguments required by the given model
        """
        transformed_views = []
        for i, view in enumerate(views):
            transformed_view = np.ma.array((view - self.view_means[i]) @ self.weights_list[i])
            transformed_views.append(transformed_view)
        return transformed_views

    def fit_transform(self, *views: Tuple[np.ndarray, ...], **kwargs):
        """
        Fits and then transforms the training data

        :param views: numpy arrays with the same number of rows (samples) separated by commas
        :param kwargs: any additional keyword arguments required by the given model
        :rtype: Tuple[np.ndarray, ...]
        """
        return self.fit(*views).transform(*views, **kwargs)

    def predict_corr(self, *views: Tuple[np.ndarray, ...], **kwargs) -> np.ndarray:
        """
        Predicts the correlation for the given data using the fit model

        :param views: numpy arrays with the same number of rows (samples) separated by commas
        :param kwargs: any additional keyword arguments required by the given model
        :return: all_corrs: an array of the pairwise correlations (k,k,self.latent_dims) where k is the number of views
        :rtype: np.ndarray
        """
        # Takes two views and predicts their out of sample correlation using trained model
        transformed_views = self.transform(*views, **kwargs)
        all_corrs = []
        for x, y in itertools.product(transformed_views, repeat=2):
            all_corrs.append(np.diag(ma.corrcoef(x.T, y.T)[:self.latent_dims, self.latent_dims:]))
        all_corrs = np.array(all_corrs).reshape((len(views), len(views), self.latent_dims))
        return all_corrs

    def demean_data(self, *views: Tuple[np.ndarray, ...]):
        """
        Removes the mean of the training data for each view and stores it

        :param views: numpy arrays with the same number of rows (samples) separated by commas
        :return: views_input: the demeaned numpy arrays to be used to fit the model
        :rtype: Tuple[np.ndarray, ...]
        """
        views_input = []
        self.view_means = []
        for view in views:
            self.view_means.append(view.mean(axis=0))
            views_input.append(view - view.mean(axis=0))
        return views_input

    def gridsearch_fit(self, *views: Tuple[np.ndarray, ...], K=None, param_candidates=None, folds: int = 5,
                       verbose: bool = False,
                       jobs: int = 0,
                       plot: bool = False):
        """
        Implements a gridsearch over the parameters in param_candidates and returns a model fit with the optimal parameters
        in cross validation (measured by sum of correlations).

        :param views: numpy arrays with the same number of rows (samples) separated by commas
        :param K: observation matrix which can be used by GCCA
        :param param_candidates:
        :param folds: number of cross-validation folds
        :param verbose: print results of training folds
        :param jobs: number of jobs. If jobs>1 then the function can use parallelism
        :param plot: produce a hyperparameter surface plot
        """
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

        cv = _CrossValidate(self, folds=folds, verbose=verbose)

        if jobs > 0:
            out = Parallel(n_jobs=jobs)(delayed(cv.score)(*views, **param_set, K=K) for param_set in param_sets)
        else:
            out = [cv.score(*views, **param_set) for param_set in param_sets]
        cv_scores, cv_stds = zip(*out)
        max_index = cv_scores.index(max(cv_scores))

        if verbose:
            print('Best score : ', max(cv_scores), flush=True)
            print('Standard deviation : ', cv_stds[max_index], flush=True)
            print(param_sets[max_index], flush=True)

        self.cv_results_table = pd.DataFrame(zip(param_sets, cv_scores, cv_stds), columns=['params', 'scores', 'std'])
        self.cv_results_table = self.cv_results_table.join(pd.json_normalize(self.cv_results_table.params))
        self.cv_results_table.drop(columns=['params'], inplace=True)

        if plot:
            cca_zoo.plot_utils.cv_plot(cv_scores, param_sets, self.__class__.__name__)

        self.set_params(**param_sets[max_index])
        self.fit(*views)
        return self

    """
    def bayes_fit(self, *views: Tuple[np.ndarray, ...], space=None, folds: int = 5, verbose=True):
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
        self.set_params(**param_sets[max_index])
        self.fit(*views)
        return self
    """


class KCCA(_CCA_Base, BaseEstimator):
    """
    :Example:

    >>> from cca_zoo.wrappers import KCCA
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> model = KCCA()
    >>> model.fit(X1,X2)
    """

    def __init__(self, latent_dims: int = 1, kernel: str = 'linear', sigma: float = 1.0, degree: int = 1, c=None):
        """
        Constructor for KCCA

        :param kernel: the kernel type 'linear', 'rbf', 'poly'
        :param sigma: sigma parameter used by sklearn rbf kernel
        :param degree: polynomial order parameter used by sklearn polynomial kernel
        :param c: regularisation between 0 (CCA) and 1 (PLS)
        """
        super().__init__(latent_dims=latent_dims)
        self.c = c
        self.kernel = kernel
        self.sigma = sigma
        self.degree = degree

    def fit(self, *views: Tuple[np.ndarray, ...], ):
        """
        Fits a KCCA model

        :param views: numpy arrays with the same number of rows (samples) separated by commas
        """
        if self.c is None:
            self.c = [0] * len(views)
        assert (len(self.c) == len(views)), 'c requires as many values as #views'
        self.model = cca_zoo.kcca.KCCA(*self.demean_data(*views), latent_dims=self.latent_dims, kernel=self.kernel,
                                       sigma=self.sigma,
                                       degree=self.degree, c=self.c)
        self.score_list = self.model.score_list
        self.train_correlations = self.predict_corr(*views)
        return self

    def transform(self, *views: Tuple[np.ndarray, ...], ):
        """
        Transforms data given a fit kCCA model

        :param views: numpy arrays with the same number of rows (samples) separated by commas
        :param kwargs: any additional keyword arguments required by the given model
        """
        transformed_views = []
        for i, view in enumerate(views):
            transformed_views.append(
                self.model.make_kernel(view - self.view_means[i], self.model.views[i]) @ self.model.alphas[i])
        return transformed_views


class MCCA(_CCA_Base, BaseEstimator):
    """
    :Example:

    >>> from cca_zoo.wrappers import MCCA
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> model = MCCA()
    >>> model.fit(X1,X2)
    """

    def __init__(self, latent_dims: int = 1, c=None):
        """
        Constructor for MCCA

        :param latent_dims: number of latent dimensions
        :param c: list of regularisation parameters for each view (between 0:CCA and 1:PLS)
        """
        super().__init__(latent_dims=latent_dims)
        self.c = c

    def fit(self, *views: Tuple[np.ndarray, ...], ):
        """
        Fits an MCCA model

        :param views: numpy arrays with the same number of rows (samples) separated by commas
        """
        if self.c is None:
            self.c = [0] * len(views)
        assert (len(self.c) == len(views)), 'c requires as many values as #views'
        views_input = self.demean_data(*views)
        all_views = np.concatenate(views_input, axis=1)
        C = all_views.T @ all_views
        # Can regularise by adding to diagonal
        D = block_diag(*[(1 - self.c[i]) * m.T @ m + self.c[i] * np.eye(m.shape[1]) for i, m in
                         enumerate(views_input)])
        C -= block_diag(*[m.T @ m for i, m in
                          enumerate(views_input)]) - D
        n = D.shape[0]
        [eigvals, eigvecs] = eigh(C, D, subset_by_index=[n - self.latent_dims, n - 1])
        idx = np.argsort(eigvals, axis=0)[::-1]
        eigvecs = eigvecs[:, idx].real
        splits = np.cumsum([0] + [view.shape[1] for view in views])
        self.eigvals = eigvals[idx].real
        self.weights_list = [eigvecs[split:splits[i + 1], :self.latent_dims] for i, split in enumerate(splits[:-1])]
        self.score_list = [view @ self.weights_list[i] for i, view in enumerate(views_input)]
        self.weights_list = [weights / np.linalg.norm(score) for weights, score in
                             zip(self.weights_list, self.score_list)]
        self.score_list = [view @ self.weights_list[i] for i, view in enumerate(views_input)]
        self.train_correlations = self.predict_corr(*views)
        return self


class GCCA(_CCA_Base, BaseEstimator):
    """
    :Example:

    >>> from cca_zoo.wrappers import GCCA
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> model = GCCA()
    >>> model.fit(X1,X2)
    """

    def __init__(self, latent_dims: int = 1, c=None, view_weights=None):
        """
        Constructor for GCCA

        :param latent_dims: number of latent dimensions
        :param c: regularisation between 0 (CCA) and 1 (PLS)
        :param view_weights: list of weights of each view
        """
        super().__init__(latent_dims=latent_dims)
        self.c = c
        self.view_weights = view_weights

    def fit(self, *views: Tuple[np.ndarray, ...], K: np.ndarray = None):
        """
        Fits a GCCA model

        :param views: numpy arrays with the same number of rows (samples) separated by commas
        :param K: observation matrix. Binary array with (k,n) dimensions where k is the number of views and n is the number of samples
        1 means the data is observed in the corresponding view and 0 means the data is unobserved in that view.
        """
        if self.c is None:
            self.c = [0] * len(views)
        assert (len(self.c) == len(views)), 'c requires as many values as #views'
        if self.view_weights is None:
            self.view_weights = [1] * len(views)
        if K is None:
            # just use identity when all rows are observed in all views.
            K = np.ones((len(views), views[0].shape[0]))
        views_input = self.demean_observed_data(*views, K=K)
        Q = []
        for i, (view, view_weight) in enumerate(zip(views_input, self.view_weights)):
            view_cov = view.T @ view
            view_cov = (1 - self.c[i]) * view_cov + self.c[i] * np.eye(view_cov.shape[0])
            Q.append(view_weight * view @ np.linalg.inv(view_cov) @ view.T)
        Q = np.sum(Q, axis=0)
        Q = np.diag(np.sqrt(np.sum(K, axis=0))) @ Q @ np.diag(np.sqrt(np.sum(K, axis=0)))
        n = Q.shape[0]
        [eigvals, eigvecs] = eigh(Q, subset_by_index=[n - self.latent_dims, n - 1])
        idx = np.argsort(eigvals, axis=0)[::-1]
        eigvecs = eigvecs[:, idx].real
        self.eigvals = eigvals[idx].real
        self.weights_list = [np.linalg.pinv(view) @ eigvecs[:, :self.latent_dims] for view in views_input]
        self.score_list = [view @ self.weights_list[i] for i, view in enumerate(views_input)]
        self.train_correlations = self.predict_corr(*views)
        return self

    def demean_observed_data(self, *views: Tuple[np.ndarray, ...], K):
        """
        Since most methods require zero-mean data, demean_data() is used to demean training data as well as to apply this
        demeaning transformation to out of sample data

        :param views: numpy arrays with the same number of rows (samples) separated by commas
        :param K: observation matrix. Binary array with (k,n) dimensions where k is the number of views and n is the number of samples
        1 means the data is observed in the corresponding view and 0 means the data is unobserved in that view.
        """
        views_input = []
        self.view_means = []
        for i, (observations, view) in enumerate(zip(K, views)):
            observed = np.where(observations == 1)[0]
            self.view_means.append(view[observed].mean(axis=0))
            view[observed] = view[observed] - self.view_means[i]
            views_input.append(np.diag(observations) @ view)
        return views_input

    def transform(self, *views: Tuple[np.ndarray, ...], K=None):
        """
        Transforms data given a fit GCCA model

        :param views: numpy arrays with the same number of rows (samples) separated by commas
        :param K: observation matrix. Binary array with (k,n) dimensions where k is the number of views and n is the number of samples
        1 means the data is observed in the corresponding view and 0 means the data is unobserved in that view.
        """
        transformed_views = []
        for i, view in enumerate(views):
            transformed_view = np.ma.array((view - self.view_means[i]) @ self.weights_list[i])
            if K is not None:
                transformed_view.mask[np.where(K[i]) == 1] = True
            transformed_views.append(transformed_view)
        return transformed_views


def _pca_data(*views: Tuple[np.ndarray, ...]):
    """
    Since most methods require zero-mean data, demean_data() is used to demean training data as well as to apply this
    demeaning transformation to out of sample data

    :param views: numpy arrays with the same number of rows (samples) separated by commas
    """
    views_U = []
    views_S = []
    views_Vt = []
    for i, view in enumerate(views):
        U, S, Vt = np.linalg.svd(view, full_matrices=False)
        views_U.append(U)
        views_S.append(S)
        views_Vt.append(Vt)
    return views_U, views_S, views_Vt


class rCCA(_CCA_Base, BaseEstimator):
    """
    :Example:

    >>> from cca_zoo.wrappers import rCCA
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> model = rCCA()
    >>> model.fit(X1,X2)
    """

    def __init__(self, latent_dims: int = 1, c=None):
        """
        Constructor for rCCA

        :param latent_dims: number of latent dimensions
        :param c: regularisation between 0 (CCA) and 1 (PLS)
        """
        super().__init__(latent_dims=latent_dims)
        self.c = c

    def fit(self, *views: Tuple[np.ndarray, ...]):
        """
        Fits a regularised CCA (canonical ridge) model

        :param views: numpy arrays with the same number of rows (samples) separated by commas
        """
        if self.c is None:
            self.c = [0] * len(views)
        assert (len(self.c) == len(views)), 'c requires as many values as #views'
        views_input = self.demean_data(*views)
        U_list, S_list, Vt_list = _pca_data(*views_input)
        if len(views) == 2:
            self.two_view_fit(U_list, S_list, Vt_list)
        else:
            self.multi_view_fit(U_list, S_list, Vt_list)
        self.score_list = [view @ self.weights_list[i] for i, view in enumerate(views_input)]
        self.train_correlations = self.predict_corr(*views)
        return self

    def two_view_fit(self, U_list, S_list, Vt_list):
        B_list = [(1 - self.c[i]) * S * S + self.c[i] for i, S in
                  enumerate(S_list)]
        R_list = [U @ np.diag(S) for U, S in zip(U_list, S_list)]
        R_12 = R_list[0].T @ R_list[1]
        M = np.diag(1 / np.sqrt(B_list[1])) @ R_12.T @ np.diag(1 / B_list[0]) @ R_12 @ np.diag(1 / np.sqrt(B_list[1]))
        n = M.shape[0]
        [eigvals, eigvecs] = eigh(M, subset_by_index=[n - self.latent_dims, n - 1])
        idx = np.argsort(eigvals, axis=0)[::-1]
        eigvecs = eigvecs[:, idx].real
        eigvals = np.real(np.sqrt(eigvals))[:self.latent_dims]
        w_y = Vt_list[1].T @ np.diag(1 / np.sqrt(B_list[1])) @ eigvecs[:, :self.latent_dims].real
        w_x = Vt_list[0].T @ np.diag(1 / B_list[0]) @ R_12 @ eigvecs[:, :self.latent_dims].real / eigvals
        self.weights_list = [w_x, w_y]

    def multi_view_fit(self, U_list, S_list, Vt_list):
        B_list = [(1 - self.c[i]) * S * S + self.c[i] for i, S in
                  enumerate(S_list)]
        D = block_diag(*[np.diag((1 - self.c[i]) * S * S + self.c[i]) for i, S in
                         enumerate(S_list)])
        C = np.concatenate([U @ np.diag(S) for U, S in zip(U_list, S_list)], axis=1)
        C = C.T @ C
        C -= block_diag(*[np.diag(S ** 2) for U, S in zip(U_list, S_list)]) - D
        n = C.shape[0]
        [eigvals, eigvecs] = eigh(C, D, subset_by_index=[n - self.latent_dims, n - 1])
        idx = np.argsort(eigvals, axis=0)[::-1]
        eigvecs = eigvecs[:, idx].real
        splits = np.cumsum([0] + [U.shape[1] for U in U_list])
        self.weights_list = [Vt.T @ np.diag(1 / np.sqrt(B)) @ eigvecs[split:splits[i + 1], :self.latent_dims] for
                             i, (split, Vt, B) in enumerate(zip(splits[:-1], Vt_list, B_list))]


class CCA(rCCA):
    """
    Implements CCA by inheriting regularised CCA with 0 regularisation
    :Example:

    >>> from cca_zoo.wrappers import CCA
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> model = CCA()
    >>> model.fit(X1,X2)
    """

    def __init__(self, latent_dims: int = 1):
        """
        Constructor for CCA

        :param latent_dims:
        """
        super().__init__(latent_dims=latent_dims, c=[0, 0])


class _Iterative(_CCA_Base):
    def __init__(self, latent_dims: int = 1, deflation='cca', max_iter=50, generalized=False,
                 initialization='unregularized', tol=1e-5):
        """
        Constructor for _Iterative

        :param latent_dims:
        :param deflation:
        :param max_iter:
        :param generalized:
        :param initialization:
        :param tol:
        """
        super().__init__(latent_dims=latent_dims)
        self.max_iter = max_iter
        self.generalized = generalized
        self.initialization = initialization
        self.tol = tol

    def fit(self, *views: Tuple[np.ndarray, ...], ):
        """
        Fits the model by running an inner loop to convergence and then using deflation (currently only supports CCA deflation)

        :param views: numpy arrays with the same number of rows (samples) separated by commas
        """
        self.set_loop_params()
        views_input = self.demean_data(*views)
        n = views_input[0].shape[0]
        p = [view.shape[1] for view in views_input]
        # list of d: p x k
        self.weights_list = [np.zeros((p_, self.latent_dims)) for p_ in p]

        # list of d: n x k
        self.score_list = [np.zeros((n, self.latent_dims)) for _ in views_input]

        residuals = copy.deepcopy(list(views_input))

        self.objective = []
        # For each of the dimensions
        for k in range(self.latent_dims):
            self.loop = self.loop.fit(*residuals)
            for i, residual in enumerate(residuals):
                self.weights_list[i][:, k] = self.loop.weights[i]
                self.score_list[i][:, k] = self.loop.scores[i]
                # TODO This is CCA deflation (https://ars.els-cdn.com/content/image/1-s2.0-S0006322319319183-mmc1.pdf)
                # but in principle we could apply any form of deflation here
                # residuals[i] = residuals[i] - np.outer(self.score_list[i][:, k], self.score_list[i][:, k]) @ residuals[
                #    i] / np.dot(self.score_list[i][:, k], self.score_list[i][:, k]).item()
                residuals[i] = self.deflate(residuals[i], self.score_list[i][:, k])
            self.objective.append(self.loop.track_objective)
        self.train_correlations = self.predict_corr(*views)
        return self

    def deflate(self, residual, score):
        """
        Deflate view residual by CCA deflation (https://ars.els-cdn.com/content/image/1-s2.0-S0006322319319183-mmc1.pdf)

        :param residual:
        :param score:

        """
        return residual - np.outer(score, score) @ residual / np.dot(score, score).item()

    @abstractmethod
    def set_loop_params(self):
        """
        

        :return:
        :rtype:
        """
        self.loop = cca_zoo.innerloop.PLSInnerLoop(max_iter=self.max_iter, generalized=self.generalized,
                                                   initialization=self.initialization)


class PLS(_Iterative):
    """
    Fits a partial least squares model with CCA deflation by NIPALS algorithm
    :Example:

    >>> from cca_zoo.wrappers import PLS
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> model = PLS()
    >>> model.fit(X1,X2)
    """

    def __init__(self, latent_dims: int = 1, max_iter=100, generalized=False, initialization='unregularized', tol=1e-5):
        """
        Constructor for PLS
        :param latent_dims: Number of latent dimensions
        :param max_iter: Maximum number of iterations
        """
        super().__init__(latent_dims=latent_dims, max_iter=max_iter, generalized=generalized,
                         initialization=initialization, tol=tol)

    def set_loop_params(self):
        self.loop = cca_zoo.innerloop.PLSInnerLoop(max_iter=self.max_iter, generalized=self.generalized,
                                                   initialization=self.initialization, tol=self.tol)


class CCA_ALS(_Iterative):
    """
    Fits a CCA model with CCA deflation by NIPALS algorithm
    :Example:

    >>> from cca_zoo.wrappers import CCA_ALS
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> model = CCA_ALS()
    >>> model.fit(X1,X2)
    """

    def __init__(self, latent_dims: int = 1, max_iter=100, generalized=False, initialization='unregularized', tol=1e-5):
        """
        Constructor for CCA_ALS
        :param latent_dims: Number of latent dimensions
        :param max_iter: Maximum number of iterations
        """
        super().__init__(latent_dims=latent_dims, max_iter=max_iter, generalized=generalized,
                         initialization=initialization, tol=tol)

    def set_loop_params(self):
        self.loop = cca_zoo.innerloop.CCAInnerLoop(max_iter=self.max_iter, generalized=self.generalized,
                                                   initialization=self.initialization, tol=self.tol)


class PMD(_Iterative, BaseEstimator):
    """
    Fits a sparse CCA model by penalized matrix decomposition
    :Example:

    >>> from cca_zoo.wrappers import PMD
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> model = PMD(c=[1,1])
    >>> model.fit(X1,X2)
    """

    def __init__(self, latent_dims: int = 1, max_iter=100, c=None, generalized=False, initialization='unregularized',
                 tol=1e-5):
        """
        Constructor for PMD
        :param latent_dims: Number of latent dimensions
        :param max_iter: Maximum number of iterations
        """
        self.c = c
        super().__init__(latent_dims=latent_dims, max_iter=max_iter, generalized=generalized,
                         initialization=initialization, tol=tol)

    def set_loop_params(self):
        self.loop = cca_zoo.innerloop.PMDInnerLoop(max_iter=self.max_iter, c=self.c, generalized=self.generalized,
                                                   initialization=self.initialization, tol=self.tol)


class ParkhomenkoCCA(_Iterative, BaseEstimator):
    """
    Fits a sparse CCA model by penalized power method
    :Example:

    >>> from cca_zoo.wrappers import ParkhomenkoCCA
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> model = ParkhomenkoCCA(c=[0.001,0.001])
    >>> model.fit(X1,X2)
    """

    def __init__(self, latent_dims: int = 1, max_iter=100, c=None, generalized=False, initialization='unregularized',
                 tol=1e-5):
        """
        Constructor for ParkhomenkoCCA
        :param latent_dims: Number of latent dimensions
        :param max_iter: Maximum number of iterations
        """
        self.c = c
        super().__init__(latent_dims=latent_dims, max_iter=max_iter, generalized=generalized,
                         initialization=initialization, tol=tol)

    def set_loop_params(self):
        self.loop = cca_zoo.innerloop.ParkhomenkoInnerLoop(max_iter=self.max_iter, c=self.c,
                                                           generalized=self.generalized,
                                                           initialization=self.initialization, tol=self.tol)


class SCCA(_Iterative, BaseEstimator):
    """
    Fits a sparse CCA model by iterative rescaled lasso regression
    :Example:

    >>> from cca_zoo.wrappers import SCCA
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> model = SCCA(c=[0.001,0.001])
    >>> model.fit(X1,X2)
    """

    def __init__(self, latent_dims: int = 1, max_iter=100, c=None, generalized=False, initialization='unregularized',
                 tol=1e-5):
        """
        Constructor for SCCA
        :param latent_dims: Number of latent dimensions
        :param max_iter: Maximum number of iterations
        """
        self.c = c
        super().__init__(latent_dims=latent_dims, max_iter=max_iter, generalized=generalized,
                         initialization=initialization, tol=tol)

    def set_loop_params(self):
        self.loop = cca_zoo.innerloop.SCCAInnerLoop(max_iter=self.max_iter, c=self.c, generalized=self.generalized,
                                                    initialization=self.initialization, tol=self.tol)


class SCCA_ADMM(_Iterative, BaseEstimator):
    """
    Fits a sparse CCA model by alternating ADMM
    :Example:

    >>> from cca_zoo.wrappers import SCCA_ADMM
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> model = SCCA_ADMM()
    >>> model.fit(X1,X2)
    """

    def __init__(self, latent_dims: int = 1, max_iter=100, c=None, mu=None, lam=None, eta=None, generalized=False,
                 initialization='unregularized', tol=1e-5):
        """
        Constructor for SCCA_ADMM
        :param latent_dims: Number of latent dimensions
        :param max_iter: Maximum number of iterations
        :param c:
        :param mu:
        :param lam:
        """
        self.c = c
        self.mu = mu
        self.lam = lam
        self.eta = eta
        super().__init__(latent_dims=latent_dims, max_iter=max_iter, generalized=generalized,
                         initialization=initialization, tol=tol)

    def set_loop_params(self):
        self.loop = cca_zoo.innerloop.ADMMInnerLoop(max_iter=self.max_iter, c=self.c, mu=self.mu, lam=self.lam,
                                                    eta=self.eta, generalized=self.generalized,
                                                    initialization=self.initialization, tol=self.tol)


class ElasticCCA(_Iterative, BaseEstimator):
    """
    Fits an elastic CCA by iterative rescaled elastic net regression
    :Example:

    >>> from cca_zoo.wrappers import ElasticCCA
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> model = ElasticCCA()
    >>> model.fit(X1,X2)
    """

    def __init__(self, latent_dims: int = 1, max_iter=100, c=None, l1_ratio=None, generalized=False,
                 initialization='unregularized', tol=1e-5, constrained=False):
        """
        Constructor for ElasticCCA
        :param latent_dims: Number of latent dimensions
        :param max_iter: Maximum number of iterations
        """
        self.c = c
        self.l1_ratio = l1_ratio
        self.constrained = constrained
        super().__init__(latent_dims=latent_dims, max_iter=max_iter, generalized=generalized,
                         initialization=initialization, tol=tol)

    def set_loop_params(self):
        self.loop = cca_zoo.innerloop.ElasticInnerLoop(max_iter=self.max_iter, c=self.c, l1_ratio=self.l1_ratio,
                                                       generalized=self.generalized, initialization=self.initialization,
                                                       tol=self.tol, constrained=self.constrained)


class TCCA(_CCA_Base):
    """
    My own port from https://github.com/rciszek/mdr_tcca

    :Example:

    >>> from cca_zoo.wrappers import TCCA
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> model = TCCA()
    >>> model.fit(X1,X2)
    """

    def __init__(self, latent_dims: int = 1, c=None):
        """
        Constructor for TCCA

        :param latent_dims:
        :param c:
        """
        super().__init__(latent_dims)
        self.c = c

    def fit(self, *views: Tuple[np.ndarray, ...], ):
        if self.c is None:
            self.c = [0] * len(views)
        assert (len(self.c) == len(views)), 'c requires as many values as #views'
        z = self.demean_data(*views)
        n = z[0].shape[0]
        covs = [(1 - self.c[i]) * view.T @ view / (1 - n) + self.c[i] * np.eye(view.shape[1]) for i, view in
                enumerate(z)]
        covs_invsqrt = [np.linalg.inv(sqrtm(cov)) for cov in covs]
        z = [z_ @ cov_invsqrt for z_, cov_invsqrt in zip(z, covs_invsqrt)]
        for i, el in enumerate(z):
            if i == 0:
                M = el
            else:
                for _ in range(len(M.shape) - 1):
                    el = np.expand_dims(el, 1)
                M = np.expand_dims(M, -1) @ el
        M = np.mean(M, 0)
        # for i, cov_invsqrt in enumerate(covs_invsqrt):
        #    M = np.tensordot(M, cov_invsqrt, axes=[[0], [0]])
        tl.set_backend('numpy')
        M_parafac = parafac(M, self.latent_dims, verbose=True)
        self.weights_list = [cov_invsqrt @ fac for i, (view, cov_invsqrt, fac) in
                             enumerate(zip(z, covs_invsqrt, M_parafac.factors))]
        self.score_list = [view @ self.weights_list[i] for i, view in enumerate(z)]
        self.weights_list = [weights / np.linalg.norm(score) for weights, score in
                             zip(self.weights_list, self.score_list)]
        self.score_list = [view @ self.weights_list[i] for i, view in enumerate(z)]
        self.train_correlations = self.predict_corr(*views)
        return self


class _CrossValidate:
    def __init__(self, model, folds: int = 5, verbose: bool = True):
        self.folds = folds
        self.verbose = verbose
        self.model = model

    def score(self, *views: Tuple[np.ndarray, ...], K=None, **cvparams):
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
            if K is not None:
                train_obs = np.delete(K, fold_inds[fold], axis=1)
                val_obs = K[:, fold_inds[fold]]
                scores[fold] = self.model.set_params(**cvparams).fit(
                    *train_sets, K=train_obs).predict_corr(
                    *val_sets).sum(axis=-1)[np.triu_indices(len(views), 1)].sum()
            else:
                scores[fold] = self.model.set_params(**cvparams).fit(
                    *train_sets).predict_corr(
                    *val_sets).sum(axis=-1)[np.triu_indices(len(views), 1)].sum()
        metric = scores.sum(axis=0) / self.folds
        std = scores.std(axis=0)
        if np.isnan(metric):
            metric = 0
        if self.verbose:
            print(cvparams)
            print(metric)
            print(std)
        return metric, std
