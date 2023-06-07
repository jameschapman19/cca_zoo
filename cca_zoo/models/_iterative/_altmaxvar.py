from typing import Iterable, Union

import numpy as np
import torch
from skprox.proximal_operators import _proximal_operators

from cca_zoo.models._iterative._base import BaseIterative, BaseLoop
from cca_zoo.utils import _process_parameter


class AltMaxVar(BaseIterative):
    def __init__(
        self,
        latent_dims=1,
        copy_data=True,
        random_state=None,
        epochs=100,
        tol=1e-3,
        proximal="L1",
        positive=False,
        tau: Union[Iterable[float], float] = None,
        proximal_params: Iterable[dict] = None,
        gamma=0.1,
        learning_rate=0.1,
        T=100,
        convergence_checking=None,
        track=None,
        verbose=False,
    ):
        super().__init__(
            latent_dims=latent_dims,
            copy_data=copy_data,
            random_state=random_state,
            tol=tol,
            epochs=epochs,
            convergence_checking=convergence_checking,
            track=track,
            verbose=verbose,
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
        learning_rate=1e-1,
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

    def objective(self, views, scores, weights) -> int:
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
        old_weights = self.weights.copy()
        self.G = self._get_target(scores)
        converged = False
        for i, view in enumerate(batch["views"]):
            t = 0
            # initialize the previous weights to None
            prev_weights = None
            while t < self.T and not converged:
                # update the weights using the gradient descent and proximal operator
                self.weights[i] -= self.learning_rate * (
                    view.T @ (view @ self.weights[i] - self.G)
                )
                self.weights[i] = self.proximal_operators[i].prox(
                    self.weights[i], self.learning_rate
                )
                # check if the weights have changed significantly from the previous iteration
                if prev_weights is not None and np.allclose(
                    self.weights[i], prev_weights
                ):
                    # if yes, set converged to True and break the loop
                    converged = True
                    break
                # update the previous weights for the next iteration
                prev_weights = self.weights[i]
                t += 1

        # if tracking or convergence_checking is enabled, compute the objective function
        if self.tracking or self.convergence_checking:
            objective = self.objective(batch["views"])
            # check that the maximum change in weights is smaller than the tolerance times the maximum absolute value of the weights
            weights_change = torch.tensor(
                np.max(
                    [
                        np.max(np.abs(old_weights[i] - self.weights[i]))
                        / np.max(np.abs(self.weights[i]))
                        for i in range(len(self.weights))
                    ]
                )
            )
            return {"loss": torch.tensor(objective), "weights_change": weights_change}
