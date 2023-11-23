# from typing import Iterable, Union
#
# import numpy as np
#
# from cca_zoo.utils import _process_parameter
#
# from ._base import _BaseIterative, BaseLoop
# from ._deflation import _DeflationMixin
#
#
# class SCCA_ADMM(_BaseIterative, _DeflationMixin):
#     r"""
#     Fits a sparse CCA model by alternating ADMM for two or more representations.
#
#     .. math::
#
#         w_{opt}=\underset{w}{\mathrm{argmax}}\{\sum_i\sum_{j\neq i} \|X_iw_i-X_jw_j\|^2 + \|w_i\|_1\}\\
#
#         \text{subject to:}
#
#         w_i^TX_i^TX_iw_i=1
#
#     Parameters
#     ----------
#     latent_dimensions : int, default=1
#         Number of latent dimensions to use in the model.
#     copy_data : bool, default=True
#         Whether to copy the data or overwrite it.
#     random_state : int, default=None
#         Random seed for initialisation.
#     deflation : str, default="cca"
#         Deflation method to use. Options are "cca" and "pls".
#     tau : float or list of floats, default=None
#         Regularisation parameter. If a single float is given, the same value is used for all representations.
#         If a list of floats is given, the values are used for each view.
#     mu : float or list of floats, default=None
#         Regularisation parameter. If a single float is given, the same value is used for all representations.
#         If a list of floats is given, the values are used for each view.
#     lam : float or list of floats, default=None
#         Regularisation parameter. If a single float is given, the same value is used for all representations.
#         If a list of floats is given, the values are used for each view.
#     eta : float or list of floats, default=None
#         Regularisation parameter. If a single float is given, the same value is used for all representations.
#         If a list of floats is given, the values are used for each view.
#     tol : float, default=1e-9
#         Tolerance for convergence.
#
#     References
#     ----------
#     Suo, Xiaotong, et al. "Sparse canonical correlation analysis." arXiv preprint arXiv:1705.10865 (2017).
#
#     Examples
#     --------
#     >>> from cca_zoo.linear import SCCA_ADMM
#     >>> import numpy as np
#     >>> rng=np.random.RandomState(0)
#     >>> X1 = rng.random((10,5))
#     >>> X2 = rng.random((10,5))
#     >>> model = SCCA_ADMM(random_state=0,tau=[1e-1,1e-1])
#     >>> model.fit((X1,X2)).score((X1,X2))
#     array([0.84348183])
#     """
#
#     def __init__(
#         self,
#         latent_dimensions: int = 1,
#         copy_data=True,
#         random_state=None,
#         tau: Union[Iterable[float], float] = None,
#         mu: Union[Iterable[float], float] = None,
#         lam: Union[Iterable[float], float] = None,
#         eta: Union[Iterable[float], float] = None,
#         initialization: Union[str, callable] = "pls",
#         tol: float = 1e-3,
#     ):
#         self.tau = tau
#         self.mu = mu
#         self.lam = lam
#         self.eta = eta
#         super().__init__(
#             latent_dimensions=latent_dimensions,
#             copy_data=copy_data,
#             initialization=initialization,
#             tol=tol,
#             random_state=random_state,
#         )
#
#     def _check_params(self):
#         self.tau = _process_parameter("tau", self.tau, 0, self.n_views_)
#         self.lam = _process_parameter("lam", self.lam, 1, self.n_views_)
#         self.eta = _process_parameter("eta", self.eta, 0, self.n_views_)
#
#     def _get_pl_module(self, weights_=None, k=None):
#         return SCCA_ADMM_PL(
#             weights_=weights_,
#             k=k,
#             tau=self.tau,
#             lam=self.lam,
#             eta=self.eta,
#             n_samples_=self.n_samples_,
#             n_views_=self.n_views_,
#         )
#
#
# class SCCA_ADMM_PL(BaseLoop):
#     def __init__(
#         self,
#         weights_,
#         k=None,
#         n_samples_=None,
#         n_views_=None,
#         tau=None,
#         eta=None,
#         lam=None,
#         mu=None,
#     ):
#         super().__init__(weights_=weights_, k=k)
#         self.eta = [np.ones(n_samples_) * eta for eta in eta]
#         self.representations = [np.ones(n_samples_)] * n_views_
#         self.mu = mu
#
#     def training_step(self, batch, batch_idx):
#         representations = batch["representations"]
#         scores = np.stack(self(representations))
#         for view_index, view in enumerate(representations):
#             targets = np.ma.array(scores, mask=False)
#             targets.mask[view_index] = True
#             gradient = representations[view_index].T @ targets.sum(axis=0).filled()
#             mu = self.mu[view_index]
#             lam = self.lam[view_index]
#             N = representations[view_index].shape[0]
#             unnorm_z = []
#             norm_eta = []
#             norm_weights = []
#             norm_proj = []
#             for _ in range(self.max_iter):
#                 # We multiply 'tau' by N in order to make regularisation match across the different sparse cca methods
#                 self.weights_[view_index] = self._prox_mu_f(
#                     self.weights_[view_index]
#                     - mu
#                     / lam
#                     * representations[view_index].T
#                     @ (
#                         representations[view_index] @ self.weights_[view_index]
#                         - self.representations[view_index]
#                         + self.eta[view_index]
#                     ),
#                     mu,
#                     gradient,
#                     N * self.tau[view_index],
#                 )
#                 unnorm_z.append(
#                     np.linalg.norm(
#                         representations[view_index] @ self.weights_[view_index]
#                         + self.eta[view_index]
#                     )
#                 )
#                 self.representations[view_index] = self._prox_lam_g(
#                     representations[view_index] @ self.weights_[view_index] + self.eta[view_index]
#                 )
#                 self.eta[view_index] = (
#                     self.eta[view_index]
#                     + representations[view_index] @ self.weights_[view_index]
#                     - self.representations[view_index]
#                 )
#                 norm_eta.append(np.linalg.norm(self.eta[view_index]))
#                 norm_proj.append(
#                     np.linalg.norm(representations[view_index] @ self.weights_[view_index])
#                 )
#                 norm_weights.append(np.linalg.norm(self.weights_[view_index], 1))
#
#     def _prox_mu_f(self, x, mu, c, tau):
#         u_update = x.copy()
#         mask_1 = x + (mu * c) > mu * tau
#         # if mask_1.sum()>0:
#         u_update[mask_1] = x[mask_1] + mu * (c[mask_1] - tau)
#         mask_2 = x + (mu * c) < -mu * tau
#         # if mask_2.sum() > 0:
#         u_update[mask_2] = x[mask_2] + mu * (c[mask_2] + tau)
#         mask_3 = ~(mask_1 | mask_2)
#         u_update[mask_3] = 0
#         return u_update
#
#     def _prox_lam_g(self, x):
#         norm = np.linalg.norm(x)
#         if norm < 1:
#             return x
#         else:
#             return x / max(1, norm)
