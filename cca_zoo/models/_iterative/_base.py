import copy
import warnings
from abc import abstractmethod
from itertools import combinations
from typing import Union, Iterable

import numpy as np
from tqdm import tqdm

from .._mcca import MCCA, KCCA
from cca_zoo.models._base import _BaseCCA
from cca_zoo.models._dummy import _DummyCCA


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
        tol: float = 1e-3,
        verbose=0,
    ):
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
        self.weights = [
            np.zeros((view.shape[1], self.latent_dims), dtype=view.dtype)
            for view in views
        ]
        initializer = _default_initializer(
            views, self.initialization, self.random_state, self.latent_dims
        )
        initializer_scores = np.stack(initializer.fit_transform(views))
        residuals = copy.deepcopy(list(views))
        self.track = {"objective": {}}
        for k in (
            tqdm(range(self.latent_dims), desc="latent dimension")
            if self.verbose > 0
            else range(self.latent_dims)
        ):
            initial_weights = [w[:, k] for w in initializer.weights]
            weights, self.track["objective"][k] = self._fit(
                residuals, initializer_scores[:, :, k], initial_weights
            )
            for i, residual in enumerate(residuals):
                self.weights[i][:, k] = weights[i]
                residuals[i] = self._deflate(residuals[i], self.weights[i][:, k])
        return self

    def _fit(self, views, scores, weights):
        objective = []
        for t in (
            tqdm(range(self.max_iter), desc="inner loop iterations")
            if self.verbose > 1
            else range(self.max_iter)
        ):
            self._initialize(views)
            scores, weights = self._update(views, scores, weights)
            if np.isnan(scores).sum() > 0:
                warnings.warn(
                    f"Some scores are nan. Usually regularisation is too high."
                )
                break
            objective.append(self._objective(views, scores, weights))
            if t > 0 and self._early_stop(objective):
                break
        return weights, objective

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
    def _update(self, views, scores, weights):
        return scores, weights

    def _objective(self, views, scores, weights) -> int:
        """
        Function used to calculate the objective function for the given. If we do not override then returns the covariance
         between projections

        :return:
        """
        # default objective is covariance
        obj = 0
        for (score_i, score_j) in combinations(scores, 2):
            obj += score_i.T @ score_j
        return -obj.item()

    def _early_stop(self, objective) -> bool:
        # Some kind of early stopping
        if (
            np.abs(objective[-1] - objective[-2])
            / np.abs(objective[-1] + objective[-2])
            < self.tol
        ):
            return True
        else:
            return False

    def _initialize(self, views):
        pass


def _default_initializer(views, initialization, random_state, latent_dims):
    if initialization == "random":
        initializer = _DummyCCA(latent_dims, random_state=random_state, uniform=False)
    elif initialization == "uniform":
        initializer = _DummyCCA(latent_dims, random_state=random_state, uniform=True)
    elif initialization == "pls":
        if sum([v.shape[0] for v in views]) < sum([v.shape[1] for v in views]):
            initializer = KCCA(latent_dims, random_state=random_state, c=1)
        else:
            initializer = MCCA(latent_dims, random_state=random_state, c=1)
    elif initialization == "cca":
        initializer = MCCA(latent_dims)
    else:
        raise ValueError(
            "Initialization {type} not supported. Pass a generator implementing this method"
        )
    return initializer


def _cosine_similarity(a, b):
    """
    Calculates the cosine similarity between vectors
    :param a: 1d numpy array
    :param b: 1d numpy array
    :return: cosine similarity
    """
    # https: // www.statology.org / cosine - similarity - python /
    return a.T @ b / (np.linalg.norm(a) * np.linalg.norm(b))
