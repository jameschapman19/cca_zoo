import copy
import warnings
from abc import abstractmethod
from itertools import combinations
from typing import Union, Iterable

import numpy as np
from sklearn.utils.validation import check_random_state
from tqdm import tqdm

from .._base import _BaseCCA
from .._multiview._mcca import MCCA, KCCA


class _BaseIterative(_BaseCCA):
    """
    A class used as the base for _iterative CCA methods i.e. those that optimize for each dimension one at a time with deflation.

    """

    def __init__(
        self,
        latent_dims: int = 1,
        scale: bool = True,
        centre=True,
        copy_data=True,
        random_state=None,
        deflation="cca",
        max_iter: int = 100,
        initialization: Union[str, callable] = "random",
        tol: float = 1e-9,
        verbose=0,
    ):
        """
        Constructor for _BaseIterative
        :param latent_dims: number of latent dimensions to fit
        :param scale: normalize variance in each column before fitting
        :param centre: demean data by column before fitting (and before transforming out of sample
        :param copy_data: If True, views will be copied; else, it may be overwritten
        :param random_state: Pass for reproducible output across multiple function calls
        :param deflation: the type of deflation.
        :param max_iter: the maximum number of iterations to perform in the inner optimization loop
        :param initialization: either string from "pls", "cca", "random", "uniform" or callable to initialize the score variables for _iterative methods
        :param tol: tolerance value used for early stopping
        """
        super().__init__(
            latent_dims=latent_dims,
            scale=scale,
            centre=centre,
            copy_data=copy_data,
            accept_sparse=["csc", "csr"],
            random_state=random_state,
        )
        self.max_iter = max_iter
        self.initialization = initialization
        self.tol = tol
        self.deflation = deflation
        self.verbose = verbose

    def fit(self, views: Iterable[np.ndarray], y=None, **kwargs):
        views = self._validate_inputs(views)
        self._check_params()
        self.weights = [np.zeros((view.shape[1], self.latent_dims)) for view in views]
        self._outer_loop(views)
        return self

    def _outer_loop(self, views):
        if isinstance(self.initialization, str):
            initializer = _default_initializer(
                views, self.initialization, self.random_state, self.latent_dims
            )
        else:
            initializer = self.initialization()
        self.track = []
        residuals = copy.deepcopy(list(views))
        for k in (
            tqdm(range(self.latent_dims), desc="latent dimension")
            if self.verbose > 0
            else range(self.latent_dims)
        ):
            self._set_loop_params()
            self.loop = self.loop._fit(residuals, initial_scores=next(initializer))
            for i, residual in enumerate(residuals):
                self.weights[i][:, k] = self.loop.weights[i].ravel()
                residuals[i] = self._deflate(residuals[i], self.weights[i][:, k])
            self.track.append(self.loop.track)
            if not self.track[-1]["converged"]:
                warnings.warn(
                    f"Inner loop {k} not converged. Increase number of iterations."
                )

    def _deflate(self, residual, weights):
        """
        Deflate view residual by CCA deflation (https://ars.els-cdn.com/content/image/1-s2.0-S0006322319319183-mmc1.pdf)

        :param residual: the current residual data matrix

        """
        score = residual @ weights
        if self.deflation == "cca":
            return residual - np.outer(score, score) @ residual / np.dot(score, score)
        elif self.deflation == "pls":
            return residual - np.outer(residual @ weights, weights)
        else:
            raise ValueError(f"deflation method {self.deflation} not implemented yet.")

    @abstractmethod
    def _set_loop_params(self):
        """
        Sets up the inner optimization loop for the method.
        """
        pass


class _BaseInnerLoop:
    def __init__(
        self,
        max_iter: int = 100,
        tol: float = 1e-9,
        random_state=None,
        verbose=0,
    ):
        self.track = {"converged": False, "objective": []}
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = check_random_state(random_state)
        self.verbose = verbose

    def _initialize(self, views):
        self.n_views = len(views)

    def _fit(self, views: np.ndarray, initial_scores):
        self.scores = initial_scores
        self.weights = [np.zeros(view.shape[1]) for view in views]
        self.n = views[0].shape[0]
        self.n_views = len(views)
        self._initialize(views)
        # Iterate until convergence
        for _ in (
            tqdm(range(self.max_iter), desc="inner loop iterations")
            if self.verbose > 1
            else range(self.max_iter)
        ):
            self._inner_iteration(views)
            if np.isnan(self.scores).sum() > 0:
                warnings.warn(
                    f"Some scores are nan. Usually regularisation is too high."
                )
                break
            self.track["objective"].append(self._objective(views))
            if _ > 1 and self._early_stop():
                self.track["converged"] = True
                break
            self.old_scores = self.scores.copy()
        return self

    def _early_stop(self) -> bool:
        # Some kind of early stopping
        if all(
            _cosine_similarity(self.scores[n], self.old_scores[n]) > (1 - self.tol)
            for n, view in enumerate(self.scores)
        ):
            return True
        else:
            return False

    @abstractmethod
    def _inner_iteration(self, views):
        pass

    def _objective(self, views) -> int:
        """
        Function used to calculate the objective function for the given. If we do not override then returns the covariance
         between projections

        :return:
        """
        # default objective is correlation
        obj = 0
        for (score_i, score_j) in combinations(self.scores, 2):
            obj += score_i.T @ score_j
        return obj.item()


def _default_initializer(views, initialization, random_state, latent_dims):
    """
    This is a generator function which generates initializations for each dimension

    :param views:
    :param initialization:
    :param random_state:
    :param latent_dims:
    :return:
    """
    sum_view_dims = sum([view.shape[1] for view in views])
    n = views[0].shape[0]
    if initialization == "random":
        while True:
            yield np.array(
                [random_state.normal(0, 1, size=(view.shape[0])) for view in views]
            )
    elif initialization == "uniform":
        while True:
            yield np.array([np.ones(view.shape[0]) for view in views])
    elif initialization == "pls":
        latent_dim = 0
        if n > sum_view_dims:
            pls_scores = MCCA(latent_dims, c=1).fit_transform(views)
        else:
            pls_scores = KCCA(latent_dims, c=1).fit_transform(views)
        while True:
            yield np.stack(pls_scores)[:, :, latent_dim]
            latent_dim += 1
    elif initialization == "cca":
        latent_dim = 0
        cca_scores = MCCA(latent_dims).fit_transform(views)
        while True:
            yield np.stack(cca_scores)[:, :, latent_dim]
            latent_dim += 1
    else:
        raise ValueError(
            "Initialization {type} not supported. Pass a generator implementing this method"
        )


def _cosine_similarity(a, b):
    """
    Calculates the cosine similarity between vectors
    :param a: 1d numpy array
    :param b: 1d numpy array
    :return: cosine similarity
    """
    # https: // www.statology.org / cosine - similarity - python /
    return a.T @ b / (np.linalg.norm(a) * np.linalg.norm(b))
