"""CCA Models"""

import copy
import itertools
from abc import abstractmethod

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

class CCA_Base(BaseEstimator):
    """
    This is the base class for the models in the cca-zoo package. Inheriting
    the class gives and

    Parameters:
    :cvar latent_dims: number of latent dimensions
    """

    @abstractmethod
    def __init__(self, latent_dims: int = 1):
        """
        :param latent_dims: number of latent dimensions
        """
        self.weights_list = None
        self.train_correlations = None
        self.latent_dims = latent_dims

    @abstractmethod
    def fit(self, *views):
        """
        The fit method takes any number of views as a numpy array along with associated parameters as a dictionary.
        Returns a fit model object which can be used to predict correlations or transform out of sample data.
        :param views: 2D numpy arrays for each view separated by comma with the same number of rows (nxp)
        :return: training data correlations and the parameters required to call other functions in the class.
        """
        pass
        return self

    def transform(self, *views, **kwargs):
        """
        The transform method takes any number of views as a numpy array. Need to have the same number of features as
        those in the views used to train the model.
        Returns the views transformed into the learnt latent space.
        :param views: numpy arrays separated by comma. Each view needs to have the same number of features as its
         corresponding view in the training data
        :return: tuple of transformed numpy arrays
        """
        transformed_views = []
        for i, view in enumerate(views):
            transformed_view = np.ma.array((view - self.view_means[i]) @ self.weights_list[i])
            transformed_views.append(transformed_view)
        return transformed_views

    def fit_transform(self, *views, **kwargs):
        """
        Apply fit and immediately transform the same data
        :param views:
        :return: tuple of transformed numpy arrays
        """
        return self.fit(*views).transform(*views, **kwargs)

    def predict_corr(self, *views, **kwargs):
        """
        :param views: numpy arrays separated by comma. Each view needs to have the same number of features as its
         corresponding view in the training data
        :return: numpy array containing correlations between each pair of views for each dimension (#views*#views*#latent_dimensions)
        """
        # Takes two views and predicts their out of sample correlation using trained model
        transformed_views = self.transform(*views, **kwargs)
        all_corrs = []
        for x, y in itertools.product(transformed_views, repeat=2):
            all_corrs.append(np.diag(ma.corrcoef(x.T, y.T)[:self.latent_dims, self.latent_dims:]))
        all_corrs = np.array(all_corrs).reshape((len(views), len(views), self.latent_dims))
        return all_corrs

    def demean_data(self, *views):
        """
        Since most methods require zero-mean data, demean_data() is used to demean training data as well as to apply this
        demeaning transformation to out of sample data
        :param views:
        :return:
        """
        views_input = []
        self.view_means = []
        for view in views:
            self.view_means.append(view.mean(axis=0))
            views_input.append(view - view.mean(axis=0))
        return views_input

    def gridsearch_fit(self, *views, K=None, param_candidates=None, folds: int = 5, verbose: bool = False,
                       jobs: int = 0,
                       plot: bool = False):
        """
        Fits the model using a user defined grid search. Returns parameters/objects that allow out of sample transformation or prediction
        Supports parallel model training with jobs>0
        :param views: numpy arrays separated by comma e.g. fit(view_1,view_2,view_3)
        :param param_candidates: dictionary containing a list for each parameter where all lists have the same length.
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
        self.set_params(**param_sets[max_index])
        self.fit(*views)
        return self
    """


class KCCA(CCA_Base, BaseEstimator):
    def __init__(self, latent_dims: int = 1, kernel: str = 'linear', sigma: float = 1.0, degree: int = 1, c=None):
        """
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

    def fit(self, *views):
        """
        The fit method takes any number of views as a numpy array along with associated parameters as a dictionary.
        Returns a fit model object which can be used to predict correlations or transform out of sample data.
        :param views: 2D numpy arrays for each view with the same number of rows (nxp)
        :return: training data correlations and the parameters required to call other functions in the class.
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

    def transform(self, *views):
        transformed_views = []
        for i, view in enumerate(views):
            transformed_views.append(
                self.model.make_kernel(view - self.view_means[i], self.model.views[i]) @ self.model.alphas[i])
        return transformed_views


class MCCA(CCA_Base, BaseEstimator):
    def __init__(self, latent_dims: int = 1, c=None):
        """
        :param latent_dims: number of latent dimensions
        :param c: list of regularisation parameters for each view (between 0:CCA and 1:PLS)
        """
        super().__init__(latent_dims=latent_dims)
        self.c = c

    def fit(self, *views):
        """
        The fit method takes any number of views as a numpy array along with associated parameters as a dictionary.
        Returns a fit model object which can be used to predict correlations or transform out of sample data.
        :param views: 2D numpy arrays for each view with the same number of rows (nxp)
        :param c: regularisation between 0 (CCA) and 1 (PLS)
        :return: training data correlations and the parameters required to call other functions in the class.
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
        self.weights_list = [weights/np.linalg.norm(score) for weights,score in zip(self.weights_list,self.score_list)]
        self.score_list = [view @ self.weights_list[i] for i, view in enumerate(views_input)]
        self.train_correlations = self.predict_corr(*views)
        return self


class GCCA(CCA_Base, BaseEstimator):
    def __init__(self, latent_dims: int = 1, c=None, view_weights=None):
        """
        :param latent_dims: number of latent dimensions
        :param c: regularisation between 0 (CCA) and 1 (PLS)
        :param view_weights: list of weights of each view
        """
        super().__init__(latent_dims=latent_dims)
        self.c = c
        self.view_weights = view_weights

    def fit(self, *views, K=None):
        """
        The fit method takes any number of views as a numpy array along with associated parameters as a dictionary.
        Returns a fit model object which can be used to predict correlations or transform out of sample data.
        :param views: 2D numpy arrays for each view with the same number of rows (nxp)
        :param K: "row selection matrix" as in Multiview LSA: Representation Learning via Generalized CCA https://www.aclweb.org/anthology/N15-1058.pdf
            binary numpy array with dimensions (#views,#rows) with one for observed and zero for not observed for that view.
        :return: training data correlations and the parameters required to call other functions in the class.
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

    def demean_observed_data(self, *views, K):
        """
        Since most methods require zero-mean data, demean_data() is used to demean training data as well as to apply this
        demeaning transformation to out of sample data
        :param views:
        :return:
        """
        views_input = []
        self.view_means = []
        for i, (observations, view) in enumerate(zip(K, views)):
            observed = np.where(observations == 1)[0]
            self.view_means.append(view[observed].mean(axis=0))
            view[observed] = view[observed] - self.view_means[i]
            views_input.append(np.diag(observations) @ view)
        return views_input

    def transform(self, *views, K=None):
        """
        The transform method takes any number of views as a numpy array. Need to have the same number of features as
        those in the views used to train the model.
        Returns the views transformed into the learnt latent space.
        :param views: numpy arrays separated by comma. Each view needs to have the same number of features as its
         corresponding view in the training data
        :return: tuple of transformed numpy arrays
        """
        transformed_views = []
        for i, view in enumerate(views):
            transformed_view = np.ma.array((view - self.view_means[i]) @ self.weights_list[i])
            if K is not None:
                transformed_view.mask[np.where(K[i]) == 1] = True
            transformed_views.append(transformed_view)
        return transformed_views


def pca_data(*views):
    """
    Since most methods require zero-mean data, demean_data() is used to demean training data as well as to apply this
    demeaning transformation to out of sample data
    :param views:
    :return:
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


class rCCA(CCA_Base, BaseEstimator):
    def __init__(self, latent_dims: int = 1, c=None):
        super().__init__(latent_dims=latent_dims)
        self.c = c

    def fit(self, *views):
        if self.c is None:
            self.c = [0] * len(views)
        assert (len(self.c) == len(views)), 'c requires as many values as #views'
        views_input = self.demean_data(*views)
        U_list, S_list, Vt_list = pca_data(*views_input)
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
    def __init__(self, latent_dims: int = 1):
        """
        Implements CCA by inheriting regularised CCA with 0 regularisation
        :param latent_dims:
        """
        super().__init__(latent_dims=latent_dims, c=[0, 0])


class Iterative(CCA_Base):
    def __init__(self, latent_dims: int = 1, deflation='cca', max_iter=50):
        super().__init__(latent_dims=latent_dims)
        self.max_iter = max_iter

    def fit(self, *views):
        """
        Fits the model for a given set of parameters (or use default values). Returns parameters/objects that allow out of sample transformation or prediction
        :param views: numpy arrays separated by comma e.g. fit(view_1,view_2,view_3)
        :return: training data correlations and the parameters required to call other functions in the class.
        """
        self.set_loop_params()
        self.outer_loop(*self.demean_data(*views))
        self.train_correlations = self.predict_corr(*views)
        return self

    def outer_loop(self, *views):
        """
        :param views: numpy arrays separated by comma. Each view needs to have the same number of features as its
         corresponding view in the training data
        :return: complete set of weights and scores required to calcuate train correlations of latent dimensions and
         transformations of out of sample data
        """
        n = views[0].shape[0]
        p = [view.shape[1] for view in views]
        # list of d: p x k
        self.weights_list = [np.zeros((p_, self.latent_dims)) for p_ in p]

        # list of d: n x k
        self.score_list = [np.zeros((n, self.latent_dims)) for _ in views]

        residuals = copy.deepcopy(list(views))

        self.objective = []
        # For each of the dimensions
        for k in range(self.latent_dims):
            self.loop = self.loop.fit(*residuals)
            for i, residual in enumerate(residuals):
                self.weights_list[i][:, k] = self.loop.weights[i]
                self.score_list[i][:, k] = self.loop.scores[i]
                # TODO This is CCA deflation (https://ars.els-cdn.com/content/image/1-s2.0-S0006322319319183-mmc1.pdf)
                # but in principle we could apply any form of deflation here
                residuals[i] = residuals[i] - np.outer(self.score_list[i][:, k], self.score_list[i][:, k]) @ residuals[
                    i] / np.dot(self.score_list[i][:, k], self.score_list[i][:, k]).item()
            self.objective.append(self.loop.track_objective)
        # can we fix the numerical instability problem?
        return self

    def deflate(self, view, residual, score):
        pass

    @abstractmethod
    def set_loop_params(self):
        self.loop = cca_zoo.innerloop.PLSInnerLoop(max_iter=self.max_iter)


class PLS(Iterative):
    def __init__(self, latent_dims: int = 1, max_iter=100):
        """
        Fits a partial least squares model with CCA deflation by NIPALS algorithm
        :param latent_dims: Number of latent dimensions
        :param max_iter: Maximum number of iterations
        """
        self.max_iter = max_iter
        super().__init__(latent_dims=latent_dims, max_iter=max_iter)

    def set_loop_params(self):
        self.loop = cca_zoo.innerloop.PLSInnerLoop(max_iter=self.max_iter)


class CCA_ALS(Iterative):
    def __init__(self, latent_dims: int = 1, max_iter=100):
        """
        Fits a CCA model with CCA deflation by NIPALS algorithm
        :param latent_dims: Number of latent dimensions
        :param max_iter: Maximum number of iterations
        """
        self.max_iter = max_iter
        super().__init__(latent_dims=latent_dims, max_iter=max_iter)

    def set_loop_params(self):
        self.loop = cca_zoo.innerloop.CCAInnerLoop(max_iter=self.max_iter)


class PMD(Iterative, BaseEstimator):
    def __init__(self, latent_dims: int = 1, max_iter=100, c=None):
        """
        Fits a sparse CCA model by penalized matrix decomposition
        :param latent_dims: Number of latent dimensions
        :param max_iter: Maximum number of iterations
        """
        self.c = c
        self.max_iter = max_iter
        super().__init__(latent_dims=latent_dims, max_iter=max_iter)

    def set_loop_params(self):
        self.loop = cca_zoo.innerloop.PMDInnerLoop(max_iter=self.max_iter, c=self.c)


class ParkhomenkoCCA(Iterative, BaseEstimator):
    def __init__(self, latent_dims: int = 1, max_iter=100, c=None):
        """
        Fits a sparse CCA model by penalization
        :param latent_dims: Number of latent dimensions
        :param max_iter: Maximum number of iterations
        """
        self.c = c
        self.max_iter = max_iter
        super().__init__(latent_dims=latent_dims, max_iter=max_iter)

    def set_loop_params(self):
        self.loop = cca_zoo.innerloop.ParkhomenkoInnerLoop(max_iter=self.max_iter, c=self.c)


class SCCA(Iterative, BaseEstimator):
    def __init__(self, latent_dims: int = 1, max_iter=100, c=None):
        """
        Fits a sparse CCA model by iterative rescaled lasso regression
        :param latent_dims: Number of latent dimensions
        :param max_iter: Maximum number of iterations
        """
        self.c = c
        self.max_iter = max_iter
        super().__init__(latent_dims=latent_dims, max_iter=max_iter)

    def set_loop_params(self):
        self.loop = cca_zoo.innerloop.SCCAInnerLoop(max_iter=self.max_iter, c=self.c)


class SCCA_ADMM(Iterative, BaseEstimator):
    def __init__(self, latent_dims: int = 1, max_iter=100, c=None, mu=None, lam=None, eta=None):
        """
        Fits a sparse CCA model by alternating ADMM
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
        self.max_iter = max_iter
        super().__init__(latent_dims=latent_dims, max_iter=max_iter)

    def set_loop_params(self):
        self.loop = cca_zoo.innerloop.ADMMInnerLoop(max_iter=self.max_iter, c=self.c, mu=self.mu, lam=self.lam,
                                                    eta=self.eta)


class ElasticCCA(Iterative, BaseEstimator):
    def __init__(self, latent_dims: int = 1, max_iter=100, c=None, l1_ratio=None):
        """
        Fits an elastic CCA by iterative rescaled elastic net regression
        :param latent_dims: Number of latent dimensions
        :param max_iter: Maximum number of iterations
        """
        self.c = c
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        super().__init__(latent_dims=latent_dims, max_iter=max_iter)

    def set_loop_params(self):
        self.loop = cca_zoo.innerloop.ElasticInnerLoop(max_iter=self.max_iter, c=self.c, l1_ratio=self.l1_ratio)


class TCCA(CCA_Base):
    """
    My own port from https://github.com/rciszek/mdr_tcca
    """
    def __init__(self, latent_dims: int = 1, c=None):
        super().__init__(latent_dims)
        self.c = c

    def fit(self, *views):
        if self.c is None:
            self.c = [0] * len(views)
        assert (len(self.c) == len(views)), 'c requires as many values as #views'
        z = self.demean_data(*views)
        n = z[0].shape[0]
        covs = [(1 - self.c[i]) * view.T @ view / (1-n) + self.c[i] * np.eye(view.shape[1]) for i, view in
                enumerate(z)]
        covs_invsqrt = [np.linalg.inv(sqrtm(cov)) for cov in covs]
        z = [z_@cov_invsqrt for z_,cov_invsqrt in zip(z,covs_invsqrt)]
        for i, el in enumerate(z):
            if i == 0:
                M = el
            else:
                for _ in range(len(M.shape) - 1):
                    el = np.expand_dims(el, 1)
                M = np.expand_dims(M, -1) @ el
        M = np.mean(M, 0)
        #for i, cov_invsqrt in enumerate(covs_invsqrt):
        #    M = np.tensordot(M, cov_invsqrt, axes=[[0], [0]])
        tl.set_backend('numpy')
        M_parafac = parafac(M, self.latent_dims, verbose=True)
        self.weights_list = [cov_invsqrt@fac for i, (view, cov_invsqrt, fac) in
                             enumerate(zip(z, covs_invsqrt, M_parafac.factors))]
        self.score_list = [view @ self.weights_list[i] for i, view in enumerate(z)]
        self.weights_list = [weights / np.linalg.norm(score) for weights, score in
                             zip(self.weights_list, self.score_list)]
        self.score_list = [view @ self.weights_list[i] for i, view in enumerate(z)]
        self.train_correlations = self.predict_corr(*views)
        return self


def slicedict(d, s):
    """
    :param d: dictionary containing e.g. c_1, c_2
    :param s:
    :return:
    """
    return {k: v for k, v in d.items() if k.startswith(s)}


class CrossValidate:
    def __init__(self, model, folds: int = 5, verbose: bool = True):
        self.folds = folds
        self.verbose = verbose
        self.model = model

    def score(self, *views, K=None, **cvparams):
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
