# from typing import Iterable, Union
#
# import numpy as np
# import torch
#
# from cca_zoo.linear._iterative._base import _BaseIterative, BaseLoop
# from cca_zoo.utils import _process_parameter
#
# #
# class AltMaxVar(_BaseIterative):
#     def __init__(
#         self,
#         latent_dimensions=1,
#         copy_data=True,
#         random_state=None,
#         epochs=100,
#         tol=1e-3,
#         proximal="L1",
#         positive=False,
#         initialization="uniform",
#         tau: Union[Iterable[float], float] = None,
#         proximal_params: Iterable[dict] = None,
#         gamma=0.1,
#         learning_rate=1e-2,
#         T=100,
#         trainer_kwargs=None,
#         convergence_checking=None,
#         track=None,
#         verbose=False,
#     ):
#         super().__init__(
#             latent_dimensions=latent_dimensions,
#             copy_data=copy_data,
#             random_state=random_state,
#             tol=tol,
#             epochs=epochs,
#             convergence_checking=convergence_checking,
#             track=track,
#             verbose=verbose,
#             trainer_kwargs=trainer_kwargs,
#             initialization=initialization,
#         )
#         self.tau = tau
#         self.proximal = proximal
#         self.proximal_params = proximal_params
#         self.gamma = gamma
#         self.learning_rate = learning_rate
#         self.T = T
#         self.positive = positive
#         # set trainer kwargs accelerator to 'cpu'
#         self.trainer_kwargs["accelerator"] = "cpu"
#
#     def _get_pl_module(self, weights_=None, k=None):
#         return AltMaxVarLoop(
#             weights_=weights_,
#             k=k,
#             gamma=self.gamma,
#             T=self.T,
#             proximal_operators=self.proximal_operators,
#             learning_rate=self.learning_rate,
#             convergence_checking=self.convergence_checking,
#             track=self.track,
#         )
#
#     def _check_params(self):
#         self.proximal = _process_parameter(
#             "proximal", self.proximal, "L1", self.n_views_
#         )
#         self.positive = _process_parameter(
#             "positive", self.positive, False, self.n_views_
#         )
#         self.tau = _process_parameter("tau", self.tau, 0, self.n_views_)
#         self.sigma = self.tau
#         self.proximal_operators = [
#             self._get_proximal(view) for view in range(self.n_views_)
#         ]
#
#     def _get_proximal(self, view):
#         if callable(self.proximal[view]):
#             params = self.proximal_params[view] or {}
#         else:
#             params = {
#                 "sigma": self.sigma[view],
#                 "positive": self.positive[view],
#             }
#         return _proximal_operators(self.proximal[view], **params)
#
#     def _more_tags(self):
#         return {"multiview": True}
#
#
# class AltMaxVarLoop(BaseLoop):
#     def __init__(
#         self,
#         weights_,
#         k=None,
#         gamma=0.1,
#         T=100,
#         proximal_operators=None,
#         learning_rate=1e-3,
#         convergence_checking=None,
#         track=None,
#     ):
#         super().__init__(
#             weights_=weights_,
#             k=k,
#             convergence_checking=convergence_checking,
#             tracking=track,
#         )
#         self.gamma = gamma
#         self.proximal_operators = proximal_operators
#         self.T = T
#         self.learning_rate = learning_rate
#
#     def forward(self, representations: list) -> list:
#         # representations detach and numpy
#         representations = [view.detach().numpy() for view in representations]
#         return [view @ weight for view, weight in zip(representations, self.weights_)]
#
#     def _get_target(self, scores):
#         if hasattr(self, "G"):
#             R = self.gamma * scores.mean(axis=0) + (1 - self.gamma) * self.G
#         else:
#             R = scores.mean(axis=0)
#         U, S, Vt = np.linalg.svd(R, full_matrices=False)
#         G = U @ Vt
#         return G / np.sqrt(np.diag(np.atleast_1d(np.cov(G, rowvar=False))))
#
#     def objective(self, representations, scores, weights_) -> int:
#         least_squares = (np.linalg.norm(scores - self.G, axis=(1, 2)) ** 2).sum()
#         regularization = np.array(
#             [self.proximal_operators[view](weights_[view]) for view in range(len(representations))]
#         ).sum()
#         return least_squares + regularization
#
#     def training_step(self, batch, batch_idx):
#         scores = np.stack(self(batch["representations"]))
#         self.G = self._get_target(scores)
#         old_weights = self.weights_.copy()
#         for i, view in enumerate(batch["representations"]):
#             view = view.detach().numpy()
#             t = 0
#             prev_weights = None
#             converged = False
#             while t < self.T and not converged:
#                 grad = view.T @ (view @ self.weights_[i] - self.G) / view.shape[0]
#                 # update the weights_ using the gradient descent and proximal operator
#                 self.weights_[i] -= self.learning_rate * grad
#                 self.weights_[i] = self.proximal_operators[i].prox(
#                     self.weights_[i], self.learning_rate
#                 )
#                 # check if the weights_ have changed significantly from the previous iteration
#                 if prev_weights is not None and np.allclose(
#                     self.weights_[i], prev_weights
#                 ):
#                     # if yes, set converged to True and break the loop
#                     converged = True
#                 # update the previous weights_ for the next iteration
#                 prev_weights = self.weights_[i].copy()
#                 t += 1
#
#         # if track or convergence_checking is enabled, compute the objective function
#         if self.tracking or self.convergence_checking:
#             objective = self.objective(batch["representations"], scores, self.weights_)
#             # check that the maximum change in weights_ is smaller than the tolerance times the maximum absolute value of the weights_
#             weights_change = torch.tensor(
#                 np.max(
#                     [
#                         np.max(np.abs(old_weights[i] - self.weights_[i]))
#                         / np.max(np.abs(self.weights_[i]))
#                         for i in range(len(self.weights_))
#                     ]
#                 )
#             )
#             return {"loss": torch.tensor(objective), "weights_change": weights_change}
#
#
# from pyproximal import (
#     L0,
#     L1,
#     L2,
#     L21,
#     Euclidean,
#     EuclideanBall,
#     L0Ball,
#     L1Ball,
#     L21_plus_L1,
#     Log,
#     Log1,
#     Nuclear,
#     NuclearBall,
# )
#
# PROXIMAL_OPERATORS = {
#     "L0": L0,
#     "L0Ball": L0Ball,
#     "L1": L1,
#     "L1Ball": L1Ball,
#     "L2": L2,
#     "L21": L21,
#     "L21_plus_L1": L21_plus_L1,
#     "Nuclear": Nuclear,
#     "NuclearBall": NuclearBall,
#     "Log": Log,
#     "Log1": Log1,
#     "Euclidean": Euclidean,
#     "EuclideanBall": EuclideanBall,
# }
#
# PROXIMAL_PARAMS = {
#     "Dummy": (),
#     "L0": frozenset(["sigma"]),
#     "L0Ball": frozenset(["radius"]),
#     "L1": frozenset(["sigma"]),
#     "L1Ball": frozenset(["n", "radius"]),
#     "L2": frozenset(["sigma"]),
#     "L21": frozenset(["ndim", "sigma"]),
#     "L21_plus_L1": frozenset(["sigma", "rho"]),
#     "TV": frozenset(["sigma", "isotropic", "dims"]),
#     "Nuclear": frozenset(["dim", "sigma"]),
#     "NuclearBall": frozenset(["dims", "radius"]),
#     "Log": frozenset(["sigma", "gamma"]),
#     "Log1": frozenset(["sigma", "delta"]),
#     "Euclidean": frozenset(["sigma"]),
#     "TVL1": frozenset(["sigma", "shape", "l1_ratio"]),
# }
#
#
# def _proximal_operators(proximal, filter_params=True, **params):
#     if proximal in PROXIMAL_OPERATORS:
#         if filter_params:
#             params = {k: params[k] for k in params if k in PROXIMAL_PARAMS[proximal]}
#         return PROXIMAL_OPERATORS[proximal](**params)
#     elif callable(proximal):
#         return proximal(**params)
