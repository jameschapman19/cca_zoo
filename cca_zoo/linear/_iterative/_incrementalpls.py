# from typing import Union
#
# import numpy as np
#
# from cca_zoo.linear._iterative._base import _BaseIterative
#
#
# class IncrementalPLS(_BaseIterative):
#     r"""
#     A class used to fit Incremental PLS
#
#     Parameters
#     ----------
#     latent_dimensions : int, optional
#         Number of latent dimensions to use, by default 1
#     copy_data : bool, optional
#         Whether to copy the data, by default True
#     random_state : int, optional
#         Random state to use, by default None
#     accept_sparse : bool, optional
#         Whether to accept sparse data, by default None
#     batch_size : int, optional
#         Batch size to use, by default 1
#     epochs : int, optional
#         Number of epochs to use, by default 1
#     simple : bool, optional
#         Whether to use the simple update, by default False
#
#     References
#     ----------
#     Arora, Raman, et al. "Stochastic optimization for PCA and PLS." 2012 50th Annual Allerton Conference on Communication, Control, and Computing (Allerton). IEEE, 2012.
#     """
#
#     def __init__(
#         self,
#         latent_dimensions: int = 1,
#         copy_data=True,
#         random_state=None,
#         accept_sparse=None,
#         batch_size=1,
#         epochs=1,
#         simple=False,
#         initialization: Union[str, callable] = "random",
#     ):
#         super().__init__(
#             latent_dimensions=latent_dimensions,
#             copy_data=copy_data,
#             accept_sparse=accept_sparse,
#             random_state=random_state,
#             batch_size=batch_size,
#             epochs=epochs,
#             initialization=initialization,
#         )
#         self.simple = simple
#
#     def _update(self, representations):
#         if not hasattr(self, "S"):
#             self.S = np.zeros(self.latent_dimensions)
#             self.count = 0
#         if self.simple:
#             self.simple_update(representations)
#         else:
#             self.incremental_update(representations)
#         return False
#
#     def incremental_update(self, representations):
#         hats = np.stack([view @ weight for view, weight in zip(representations, self.weights_)])
#         orths = [
#             view - hat @ weight.T
#             for view, weight, hat in zip(representations, self.weights_, hats)
#         ]
#         self.incrsvd(hats, orths)
#
#     def simple_update(self, representations):
#         if not hasattr(self, "M"):
#             self.M = np.zeros((representations[0].shape[1], representations[1].shape[1]))
#         self.M = (
#             representations[0].T @ representations[1]
#             + self.weights_[0] @ np.diag(self.S) @ self.weights_[1].T
#         )
#         U, S, Vt = np.linalg.svd(self.M)
#         self.weights_[0] = U[:, : self.latent_dimensions]
#         self.weights_[1] = Vt.T[:, : self.latent_dimensions]
#         self.S = S[: self.latent_dimensions]
#
#     def incrsvd(self, hats, orths):
#         Q = np.vstack(
#             (
#                 np.hstack(
#                     (
#                         np.diag(self.S) + hats[0].T @ hats[1],
#                         np.linalg.norm(orths[1], axis=1).T * hats[0].T,
#                     )
#                 ),
#                 np.hstack(
#                     (
#                         (np.linalg.norm(orths[0], axis=1).T * hats[1].T).T,
#                         np.atleast_2d(
#                             np.linalg.norm(orths[0], axis=1, keepdims=True)
#                             @ np.linalg.norm(orths[1], axis=1, keepdims=True).T
#                         ),
#                     )
#                 ),
#             )
#         )
#         U, S, Vt = np.linalg.svd(Q)
#         self.weights_[0] = (
#             np.hstack((self.weights_[0], orths[0].T / np.linalg.norm(orths[0])))
#             @ U[:, : self.latent_dimensions]
#         )
#         self.weights_[1] = (
#             np.hstack((self.weights_[1], orths[1].T / np.linalg.norm(orths[1])))
#             @ Vt.T[:, : self.latent_dimensions]
#         )
#         self.S = S[: self.latent_dimensions]
