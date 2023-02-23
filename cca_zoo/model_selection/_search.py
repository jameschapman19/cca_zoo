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
from sklearn.model_selection._search import BaseSearchCV, ParameterSampler
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


class GridSearchCV(BaseSearchCV):
    """

    :Example:
    >>> from cca_zoo.model_selection import GridSearchCV
    >>> from cca_zoo.models import MCCA
    >>> X1 = [[0, 0, 1], [1, 0, 0], [2, 2, 2], [3, 5, 4]]
    >>> X2 = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
    >>> X3 = [[0, 1, 0], [1, 9, 0], [4, 3, 3,], [12, 8, 10]]
    >>> model = MCCA()
    >>> params = {'c': [[0.1, 0.2], [0.3, 0.4], 0.1]}
    >>> GridSearchCV(model,param_grid=params, cv=3).fit([X1,X2,X3]).best_estimator_.c
    [0.1, 0.3, 0.1]

    :Notes:

    The parameters selected are those that maximize the score of the left out
    data, unless an explicit score is passed in which case it is used instead.
    If `n_jobs` was set to a value higher than one, the data is copied for each
    point in the grid (and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.
    """

    _required_parameters = ["estimator", "param_grid"]

    def __init__(
        self,
        estimator,
        param_grid,
        *,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=False,
    ):
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )
        if not isinstance(param_grid, ParameterGrid):
            self.param_grid = param2grid(param_grid)
        else:
            self.param_grid = param_grid

    def _run_search(self, evaluate_candidates):
        """Search all candidates in param_grid"""
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
        self = BaseSearchCV.fit(self, np.hstack(X), y=None, groups=None, **fit_params)
        self.best_estimator_ = self.best_estimator_["estimator"]
        self.best_params_ = {
            key[len("estimator__") :]: val for key, val in self.best_params_.items()
        }
        return self


class RandomizedSearchCV(BaseSearchCV):
    """

    :Example:
    >>> from cca_zoo.model_selection import RandomizedSearchCV
    >>> from cca_zoo.models import MCCA
    >>> from sklearn._utils.fixes import loguniform
    >>> X1 = [[0, 0, 1], [1, 0, 0], [2, 2, 2], [3, 5, 4]]
    >>> X2 = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
    >>> X3 = [[0, 1, 0], [1, 9, 0], [4, 3, 3,], [12, 8, 10]]
    >>> model = MCCA()
    >>> params = {'c': [loguniform(1e-4, 1e0), loguniform(1e-4, 1e0), [0.1]]}
    >>> def scorer(estimator, views):
    ...    scores = estimator.score(views)
    ...    return np.mean(scores)
    >>> RandomizedSearchCV(model,param_distributions=params, cv=3, scoring=scorer,n_iter=10).fit([X1,X2,X3]).n_iter
    10

    :Notes:

    The parameters selected are those that maximize the score of the held-out
    data, according to the scoring parameter.
    If `n_jobs` was set to a value higher than one, the data is copied for each
    parameter setting(and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.
    """

    _required_parameters = ["estimator", "param_distributions"]

    def __init__(
        self,
        estimator,
        param_distributions,
        *,
        n_iter=10,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        random_state=None,
        error_score=np.nan,
        return_train_score=False,
    ):
        self.param_distributions = {
            f"estimator__{key}": val for key, val in param_distributions.items()
        }
        self.n_iter = n_iter
        self.random_state = random_state
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )

    def _run_search(self, evaluate_candidates):
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
        self = BaseSearchCV.fit(self, np.hstack(X), y=y, groups=groups, **fit_params)
        self.best_estimator_ = self.best_estimator_["estimator"]
        self.best_params_ = {
            key[len("estimator__") :]: val for key, val in self.best_params_.items()
        }
        return self
