from typing import Union, Iterable

import numpy as np
from skprox.proximal_operators import _proximal_operators

from cca_zoo.models._iterative._base import BaseIterative, BaseLoop
from cca_zoo.utils import _process_parameter


class AltMaxVar(BaseIterative):
    def __init__(
        self,
        latent_dims=1,
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

    def _get_module(self, weights=None, k=None):
        return AltMaxVarLoop(
            weights=weights,
            k=k,
            gamma=self.gamma,
            T=self.T,
            learning_rate=self.learning_rate,
            proximal_operators=self.proximal_operators,
        )

    def _check_params(self):
        self.proximal = _process_parameter(
            "proximal", self.proximal, "L1", self.n_views_
        )
        self.positive = _process_parameter(
            "positive", self.positive, False, self.n_views_
        )
        self.tau = _process_parameter("tau", self.tau, 0, self.n_views_)
        self.sigma = self.tau
        self.proximal_operators = [
            self._get_proximal(view) for view in range(self.n_views_)
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

    def _more_tags(self):
        return {"multiview": True}


class AltMaxVarLoop(BaseLoop):
    def __init__(
        self,
        weights,
        k=None,
        gamma=0.1,
        T=100,
        learning_rate=1e-3,
        proximal_operators=None,
    ):
        super().__init__(weights, k)
        self.gamma = gamma
        self.proximal_operators = proximal_operators
        self.T = T
        self.learning_rate = learning_rate

    def _get_target(self, scores):
        if hasattr(self, "G"):
            R = self.gamma * scores.mean(axis=0) + (1 - self.gamma) * self.G
        else:
            R = scores.mean(axis=0)
        U, S, Vt = np.linalg.svd(R, full_matrices=False)
        G = U @ Vt
        return G

    def _objective(self, views, scores, weights) -> int:
        least_squares = (np.linalg.norm(scores - self.G, axis=(1, 2)) ** 2).sum()
        regularization = np.array(
            [
                self.proximal_operators[view](weights[view])
                for view in range(self.n_views)
            ]
        ).sum()
        return least_squares + regularization

    def training_step(self, batch, batch_idx):
        scores = np.stack(self(batch["views"]))
        self.G = self._get_target(scores)
        converged = False
        t = 0
        for view in range(len(batch["views"])):
            while t < self.T and not converged:
                self.weights[view] -= self.learning_rate * (
                    batch["views"][view].T
                    @ (batch["views"][view] @ self.weights[view] - self.G)
                )
                self.weights[view] = self.proximal_operators[view].prox(
                    self.weights[view], self.learning_rate
                )
                t += 1
                converged = (
                    np.linalg.norm(self.weights[view] - self.weights[view]) < 1e-6
                )
