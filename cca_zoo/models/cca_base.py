import itertools
from abc import abstractmethod
from typing import List, Union, Dict, Any, Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin
from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.utils.validation import check_random_state, check_is_fitted

import cca_zoo.data
import cca_zoo.models.innerloop
import cca_zoo.utils.plotting
from cca_zoo.utils import check_views


class _CCA_Base(BaseEstimator, MultiOutputMixin, RegressorMixin):
    """
    A class used as the base for methods in the package. Allows methods to inherit fit_transform, predict_corr, and gridsearch_fit
    when only fit (and transform where it is different to the default) is provided.

    :param latent_dims: number of latent dimensions to fit
    :param scale: normalize variance in each column before fitting
    """

    def __init__(self, latent_dims: int = 1, scale=True, centre=True, copy_data=True, accept_sparse=False,
                 random_state: Union[int, np.random.RandomState] = None):
        """
        Constructor for _CCA_Base

        :param latent_dims: number of latent dimensions to fit
        :param scale: normalize variance in each column before fitting
        :param centre: demean data by column before fitting (and before transforming out of sample
        :param copy_data: If True, X will be copied; else, it may be overwritten
        :param accept_sparse: Whether model can take sparse data as input
        :param random_state: Pass for reproducible output across multiple function calls
        """
        self.latent_dims = latent_dims
        self.scale = scale
        self.centre = centre
        self.copy_data = copy_data
        self.accept_sparse = accept_sparse
        self.random_state = check_random_state(random_state)
        self.n_views = None

    @abstractmethod
    def fit(self, *views: np.ndarray):
        """
        Fits a given model

        :param views: numpy arrays with the same number of rows (samples) separated by commas
        """
        raise NotImplementedError

    def transform(self, *views: np.ndarray, **kwargs):
        """
        Transforms data given a fit model

        :param views: numpy arrays with the same number of rows (samples) separated by commas
        :param kwargs: any additional keyword arguments required by the given model
        """
        check_is_fitted(self, attributes=['weights'])
        views = check_views(*views, copy=self.copy_data, accept_sparse=self.accept_sparse)
        views = self._centre_scale_transform(*views)
        transformed_views = []
        for i, (view) in enumerate(views):
            transformed_view = view @ self.weights[i]
            transformed_views.append(transformed_view)
        return transformed_views

    def fit_transform(self, *views: np.ndarray, **kwargs):
        """
        Fits and then transforms the training data

        :param views: numpy arrays with the same number of rows (samples) separated by commas
        :param kwargs: any additional keyword arguments required by the given model
        :rtype: np.ndarray
        """
        return self.fit(*views, **kwargs).transform(*views)

    def loadings(self, *views: np.ndarray, **kwargs):
        """
        Returns the model loadings for each view for the given data

        :param views: numpy arrays with the same number of rows (samples) separated by commas
        :param kwargs: any additional keyword arguments required by the given model
        """
        transformed_views = self.transform(*views, **kwargs)
        views = self._centre_scale_transform(*views)
        loadings = [view.T @ transformed_view for view, transformed_view in zip(views, transformed_views)]
        return loadings

    def _centre_scale(self, *views: np.ndarray):
        """
        Removes the mean of the training data and standardizes for each view and stores mean and standard deviation during training

        :param views: numpy arrays with the same number of rows (samples) separated by commas
        :return: train_views: the demeaned numpy arrays to be used to fit the model
        """

        self.view_means = []
        self.view_stds = []
        transformed_views = []
        for view in views:
            if issparse(view):
                view_mean, view_std = mean_variance_axis(view, axis=0)
                self.view_means.append(view_mean)
                self.view_stds.append(view_std)
                view = view - self.view_means[-1]
                view = view / self.view_stds[-1]
            else:
                if self.centre:
                    view_mean = view.mean(axis=0)
                    self.view_means.append(view_mean)
                    view = view - self.view_means[-1]
                if self.scale:
                    view_std = view.std(axis=0, ddof=1)
                    view_std[view_std == 0.0] = 1.0
                    self.view_stds.append(view_std)
                    view = view / self.view_stds[-1]
            transformed_views.append(view)
        return transformed_views

    def _centre_scale_transform(self, *views: np.ndarray):
        """
        Removes the mean and standardizes each view based on the mean and standard deviation of the training data

        :param views: numpy arrays with the same number of rows (samples) separated by commas
        """
        views = check_views(*views, copy=self.copy_data, accept_sparse=self.accept_sparse)
        if self.centre:
            views = [view - mean for view, mean in zip(views, self.view_means)]
        if self.scale:
            views = [view / std for view, std in zip(views, self.view_stds)]
        return views

    def correlations(self, *views: Iterable[np.ndarray], **kwargs):
        """
        Predicts the correlation for the given data using the fit model

        :param views: numpy arrays with the same number of rows (samples) separated by commas
        :param kwargs: any additional keyword arguments required by the given model
        :return: all_corrs: an array of the pairwise correlations (k,k,self.latent_dims) where k is the number of views
        :rtype: np.ndarray
        """
        transformed_views = self.transform(*views, **kwargs)
        all_corrs = []
        for x, y in itertools.product(transformed_views, repeat=2):
            all_corrs.append(np.diag(np.corrcoef(x.T, y.T)[:self.latent_dims, self.latent_dims:]))
        all_corrs = np.array(all_corrs).reshape((len(views), len(views), self.latent_dims))
        return all_corrs

    def gridsearch_fit(self, *views: np.ndarray, K=None, param_candidates: Dict[str, List[Any]] = None,
                       folds: int = 5,
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
        if param_candidates is None:
            param_candidates = {}
        if verbose:
            print('cross validation', flush=True)
            print('number of folds: ', folds, flush=True)

        # Set up an array for each set of hyperparameters
        if len(param_candidates) == 0:
            raise ValueError('No param_candidates was supplied.')

        param_names = list(param_candidates.keys())
        param_values = list(param_candidates.values())
        param_combinations = list(itertools.product(*param_values))

        param_sets = []
        for param_set in param_combinations:
            param_dict = {}
            for i, param_name in enumerate(param_names):
                param_dict[param_name] = param_set[i]
            param_sets.append(param_dict)

        cv = _CrossValidate(self, folds=folds, verbose=verbose, random_state=self.random_state)

        if jobs > 0:
            out = Parallel(n_jobs=jobs)(delayed(cv.score)(*views, **param_set, K=K) for param_set in param_sets)
        else:
            out = [cv.score(*views, **param_set) for param_set in param_sets]
        cv_scores = np.array(out)
        max_index = np.argmax(cv_scores.mean(axis=1))

        if verbose:
            print('Best score : ', cv_scores[max_index].mean(), flush=True)
            print('Standard deviation : ', cv_scores[max_index].std(), flush=True)
            print(param_sets[max_index], flush=True)

        self.cv_results_table = pd.DataFrame(zip(param_sets), columns=['params'])
        self.cv_results_table[[f'fold_{f}' for f in range(folds)]] = cv_scores
        self.cv_results_table = self.cv_results_table.join(pd.json_normalize(self.cv_results_table.params))
        self.cv_results_table.drop(columns=['params'], inplace=True)

        if plot:
            hyperparameter_plot = cca_zoo.utils.plotting.cv_plot(cv_scores.mean(axis=1), param_sets,
                                                                 self.__class__.__name__)

        self.set_params(**param_sets[max_index])
        self.fit(*views)
        return self

    def score(self, *views, **kwargs):
        # by default return the average pairwise correlation in each dimension (for 2 views just the correlation)
        pair_corrs = self.correlations(*views, **kwargs)
        # sum all the pairwise correlations for each dimension. Subtract the self correlations. Divide by the number of views. Gives average correlation
        dim_corrs = (pair_corrs.sum(axis=tuple(range(pair_corrs.ndim - 1))) - len(views)) / (
                len(views) ** 2 - len(views))
        return dim_corrs

    """
    def bayes_fit(self, *views: np.ndarray, space=None, folds: int = 5, verbose=True):
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


class _CrossValidate:
    """
    Base class used for cross validation
    """

    def __init__(self, model, folds: int = 5, verbose: bool = True, random_state=None):
        self.folds = folds
        self.verbose = verbose
        self.model = model
        self.random_state = check_random_state(random_state)

    def score(self, *views: np.ndarray, K=None, **cvparams):
        scores = np.zeros(self.folds)
        inds = np.arange(views[0].shape[0])
        self.random_state.shuffle(inds)
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
                    *train_sets, K=train_obs).score(
                    *val_sets).sum()
            else:
                self.model.set_params(**cvparams).fit(
                    *train_sets)
                scores[fold] = self.model.score(
                    *val_sets).sum()
        scores[np.isnan(scores)] = 0
        std = scores.std(axis=0)
        if self.verbose:
            print(cvparams)
            print(scores.sum(axis=0) / self.folds)
            print(std)
        return scores
