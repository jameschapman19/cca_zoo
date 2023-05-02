import copy
from typing import Union, Iterable

import numpy as np
from skprox.proximal_operators import _proximal_operators

from cca_zoo.models._iterative._base import _BaseIterative, _default_initializer
from cca_zoo.utils import _process_parameter


class AltMaxVar(_BaseIterative):
    def __init__(
        self,
        latent_dims=1,
        scale=True,
        centre=True,
        copy_data=True,
        random_state=None,
        tol=1e-3,
        proximal="L1",
        positive=False,
        tau: Union[Iterable[float], float] = None,
        proximal_params: Iterable[dict] = None,
        gamma=0.1,
        learning_rate=0.001,
        T=100,
    ):
        super().__init__(
            latent_dims=latent_dims,
            scale=scale,
            centre=centre,
            copy_data=copy_data,
            random_state=random_state,
            tol=tol,
        )
        self.tau = tau
        self.proximal = proximal
        self.proximal_params = proximal_params
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.T = T
        self.positive = positive

    def fit(self, views: Iterable[np.ndarray], y=None, **kwargs):
        views = self._validate_inputs(views)
        self._check_params()
        self.weights = [np.zeros((view.shape[1], self.latent_dims)) for view in views]
        initializer = _default_initializer(
            views, self.initialization, self.random_state, self.latent_dims
        )
        initializer_scores = np.stack(initializer.fit_transform(views))
        residuals = copy.deepcopy(list(views))
        self.track = {"objective": {}}
        initial_weights = initializer.weights
        self.weights, self.track["objective"] = self._fit(
            residuals, initializer_scores, initial_weights
        )
        return self

    def _objective(self, views, scores, weights) -> int:
        least_squares = (np.linalg.norm(scores - self.G, axis=(1, 2)) ** 2).sum()
        regularization = np.array(
            [
                self.proximal_operator[view](weights[view])
                for view in range(self.n_views)
            ]
        ).sum()
        return least_squares + regularization

    def _update(self, views, scores, weights):
        self.G = self._get_target(scores)
        converged = False
        t = 0
        for view in range(self.n_views):
            while t < self.T and not converged:
                weights[view] -= self.learning_rate * (
                    views[view].T @ (views[view] @ weights[view] - self.G)
                )
                weights[view] = self.proximal_operator[view].prox(
                    weights[view], self.learning_rate
                )
                t += 1
                converged = np.linalg.norm(weights[view] - self.weights[view]) < 1e-6
                scores[view] = views[view] @ weights[view]
        return scores, weights

    def _check_params(self):
        self.proximal = _process_parameter(
            "proximal", self.proximal, "L1", self.n_views
        )
        self.positive = _process_parameter(
            "positive", self.positive, False, self.n_views
        )
        self.tau = _process_parameter("tau", self.tau, 0, self.n_views)
        self.sigma = self.tau
        self.proximal_operator = [
            self._get_proximal(view) for view in range(self.n_views)
        ]

    def _get_proximal(self, view):
        if callable(self.proximal[view]):
            params = self.proximal_params[view] or {}
        else:
            params = {
                "sigma": self.sigma[view],
                "positive": self.positive[view],
            }
        return _proximal_operators(self.proximal[view], **params)

    def _get_target(self, scores):
        if hasattr(self, "G"):
            R = self.gamma * scores.mean(axis=0) + (1 - self.gamma) * self.G
        else:
            R = scores.mean(axis=0)
        U, S, Vt = np.linalg.svd(R, full_matrices=False)
        G = U @ Vt
        return G

    def _more_tags(self):
        return {"multiview": True}
