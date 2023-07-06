# Author: James Chapman
# This code heavily leans on the scikit-learn original
# Original Authors:
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>,
#         Gael Varoquaux <gael.varoquaux@normalesup.org>
#         Andreas Mueller <amueller@ais.uni-bonn.de>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Raghav RV <rvraghav93@gmail.com>

import itertools
from typing import Iterable

import numpy as np
from mvlearn.compose import SimpleSplitter
from sklearn import clone
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection._search import ParameterSampler
from sklearn.model_selection._search import GridSearchCV as GridSearchCV_sklearn
from sklearn.model_selection._search import RandomizedSearchCV as RandomizedSearchCV_sklearn
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state


def param2grid(params):
    """
    Converts parameters with a list for each view into a scikit-learn friendly form
    Parameters

    :param params : a dictionary of parameters where some parameters may contain a list of lists (one list for each 'view')
    Returns : a parameter grid in the form expected by scikit-learn where each element is a single candidate
    (a single value or a list with one value for each view)

    Example
    -------
    >>> params = {'regs': [[1, 2], [3, 4]]}
    >>> param2grid(params)
    {'regs': [[1, 3], [1, 4], [2, 3], [2, 4]]}
    """
    params = params.copy()
    for k, v in params.items():
        if any([isinstance(v_, list) for v_ in v]):
            # itertools expects all lists to perform product
            v = [[v_] if not isinstance(v_, list) else v_ for v_ in v]
            params[k] = list(map(list, itertools.product(*v)))
    return ParameterGrid(params)


class ParameterSampler_(ParameterSampler):
    """
    Generator on parameters sampled from given distributions.
    Non-deterministic iterable over random candidate combinations for hyper-
    parameter search. If all parameters are presented as a list,
    sampling without replacement is performed. If at least one parameter
    is given as a distribution, sampling with replacement is used.
    It is highly recommended to use continuous distributions for continuous
    parameters.
    Read more in the :ref:`User Guide <grid_search>`.

    :param param_distributions: dict
        Dictionary with parameters names (`str`) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.
        If a list of dicts is given, first a dict is sampled uniformly, and
        then a parameter is sampled using that dict as above.
    :param n_iter: int
        Number of parameter settings that are produced.
    :param random_state: int, RandomState instance or None, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        Pass an int for reproducible output across multiple
        function calls.
        See :term:`Glossary <random_state>`.

    :Example:
    >>> from sklearn.model_selection import ParameterSampler
    >>> from scipy.stats.distributions import expon
    >>> import numpy as np
    >>> rng = np.random.RandomState(0)
    >>> param_grid = {'a':[1, 2], 'b': expon()}
    >>> param_list = list(ParameterSampler(param_grid, n_iter=4,
    ...                                    random_state=rng))
    >>> rounded_list = [dict((k, round(v, 6)) for (k, v) in d.items())
    ...                 for d in param_list]
    >>> rounded_list == [{'b': 0.89856, 'a': 1},
    ...                  {'b': 0.923223, 'a': 1},
    ...                  {'b': 1.878964, 'a': 2},
    ...                  {'b': 1.038159, 'a': 2}]
    True
    """

    def __iter__(self):
        rng = check_random_state(self.random_state)
        for _ in range(self.n_iter):
            dist = rng.choice(self.param_distributions)
            # Always sort the keys of a dictionary, for reproducibility
            items = sorted(dist.items())
            params = dict()
            for k, v in items:
                # if value is an iterable then either the elements are the distribution or each element is a distribution
                # for each view.
                if isinstance(v, Iterable):
                    # if each element is a distribution for each view (i.e. it is a non-string Iterable) then call return_param for each view
                    if any(
                        [
                            (isinstance(v_, Iterable) and not isinstance(v_, str))
                            or hasattr(v_, "rvs")
                            for v_ in v
                        ]
                    ):
                        params[k] = [self.return_param(v_) for v_ in v]
                    # if the parameter is shared across views then the list will just contain non-iterable values
                    else:
                        params[k] = self.return_param(v)
                # if value is not iterable then it is either a distribution or a value in which case call return param on it.
                else:
                    params[k] = self.return_param(v)
            yield params

    def return_param(self, v):
        rng = check_random_state(self.random_state)
        if hasattr(v, "rvs"):
            param = v.rvs(random_state=rng)
        elif isinstance(v, Iterable) and not isinstance(v, str):
            param = v[rng.randint(len(v))]
        else:
            param = v
        return param

    def __len__(self):
        """Number of points that will be sampled."""
        return self.n_iter


class GridSearchCV(GridSearchCV_sklearn):

    def _run_search(self, evaluate_candidates):
        """Search all candidates in param_grid"""
        if not isinstance(self.param_grid, ParameterGrid):
            param_grid = param2grid(self.param_grid)
        else:
            param_grid = self.param_grid
        param_grid.param_grid = [
            {f"estimator__{key}": val for key, val in subgrid.items()}
            for subgrid in param_grid.param_grid
        ]
        evaluate_candidates(param_grid)

    def fit(self, X, y=None, *, groups=None, **fit_params):
        self.estimator = Pipeline(
            [
                ("splitter", SimpleSplitter([X_.shape[1] for X_ in X])),
                ("estimator", clone(self.estimator)),
            ]
        )
        super().fit(np.hstack(X), y=y, groups=groups, **fit_params)
        self.best_estimator_ = self.best_estimator_["estimator"]
        self.best_params_ = {
            key[len("estimator__"):]: val for key, val in self.best_params_.items()
        }
        return self


class RandomizedSearchCV(RandomizedSearchCV_sklearn):

    def _run_search(self, evaluate_candidates):
        self.param_distributions = {
            f"estimator__{key}": val for key, val in self.param_distributions.items()
        }
        """Search n_iter candidates from param_distributions"""
        evaluate_candidates(
            ParameterSampler_(
                self.param_distributions, self.n_iter, random_state=self.random_state
            )
        )

    def fit(self, X, y=None, *, groups=None, **fit_params):
        self.estimator = Pipeline(
        [
            ("splitter", SimpleSplitter([X_.shape[1] for X_ in X])),
            ("estimator", clone(self.estimator)),
        ]
        )
        super().fit(np.hstack(X), y=y, groups=groups, **fit_params)
        self.best_estimator_ = self.best_estimator_["estimator"]
        self.best_params_ = {
            key[len("estimator__"):]: val for key, val in self.best_params_.items()
        }
        return self
