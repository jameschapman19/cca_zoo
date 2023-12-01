import itertools
from typing import Iterable

import numpy as np
from sklearn import clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection._search import (
    BaseSearchCV,
)
from sklearn.model_selection._search import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state

from cca_zoo._utils._splitter import SimpleSplitter


def param2grid(params):
    params = {
        k: list(v) if (hasattr(v, "__iter__") and not isinstance(v, str)) else [v]
        for k, v in params.items()
    }
    for k, v in params.items():
        if any([hasattr(v_, "__iter__") and not isinstance(v_, str) for v_ in v]):
            params[k] = list(map(list, itertools.product(*v)))
    return ParameterGrid(params)


class ParameterSampler_(ParameterSampler):
    def __iter__(self):
        rng = check_random_state(self.random_state)
        for _ in range(self.n_iter):
            dist = rng.choice(self.param_distributions)
            # Always sort the keys of a dictionary, for reproducibility
            items = sorted(dist.items())
            params = dict()
            for k, v in items:
                # if v is iterable, then list comprehension else v
                if isinstance(v, Iterable) and not isinstance(v, str):
                    # use list comprehension to handle different types of values
                    params[k] = (
                        [self.return_param(v_) for v_ in v]
                        if isinstance(v, Iterable)
                        else self.return_param(v)
                    )
                else:
                    params[k] = self.return_param(v)
            yield params

    def return_param(self, v):
        rng = check_random_state(self.random_state)
        # use ternary operator to handle different types of values
        param = (
            v.rvs(random_state=rng)
            if hasattr(v, "rvs")
            else (
                v[rng.randint(len(v))]
                if isinstance(v, Iterable) and not isinstance(v, str)
                else v
            )
        )
        return param

    def __len__(self):
        """Number of points that will be sampled."""
        return self.n_iter


class BaseSearchCV(BaseSearchCV):
    def fit(self, views, y=None, *, groups=None, **fit_params):
        self.estimator = Pipeline(
            [
                ("splitter", SimpleSplitter([view.shape[1] for view in views])),
                ("estimator", clone(self.estimator)),
            ]
        )
        super().fit(np.hstack(views), y=y, groups=groups, **fit_params)
        self.estimator = self.estimator[1]
        self.best_estimator_ = self.best_estimator_[1]
        self.best_params_ = {
            key.split("estimator__")[1]: val for key, val in self.best_params_.items()
        }
        return self


class GridSearchCV(GridSearchCV, BaseSearchCV):
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


class RandomizedSearchCV(RandomizedSearchCV, BaseSearchCV):
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
