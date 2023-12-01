from typing import Union, Iterable

import numpy as np

from cca_zoo._utils._checks import _process_parameter
from cca_zoo._utils._cross_correlation import cross_cov
from cca_zoo.linear._iterative._base import _BaseIterative
from cca_zoo.linear._iterative._deflation import _DeflationMixin
from cca_zoo.linear._search import _delta_search
from cca_zoo.linear._search import support_threshold


class SCCA_Span(_DeflationMixin, _BaseIterative):
    r"""
    Fits a Sparse _CCALoss model using SpanCCA.

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{\sum_i\sum_{j\neq i} \|X_iw_i-X_jw_j\|^2 + \text{l1_ratio}\|w_i\|_1\}\\

        \text{subject to:}

        w_i^TX_i^TX_iw_i=1

    References
    ----------
    Asteris, Megasthenis, et al. "A simple and provable algorithm for sparse diagonal _CCALoss." International Conference on Machine Learning. PMLR, 2016.
    """

    def __init__(
        self,
        latent_dimensions: int = 1,
        epochs: int = 100,
        copy_data=True,
        initialization: str = "pls",
        tol: float = 1e-3,
        regularisation="l0",
        tau: Union[Iterable[Union[float, int]], Union[float, int]] = None,
        rank=1,
        positive: Union[Iterable[bool], bool] = None,
        random_state=None,
        verbose=True,
        early_stopping=False,
    ):
        super().__init__(
            latent_dimensions=latent_dimensions,
            epochs=epochs,
            copy_data=copy_data,
            initialization=initialization,
            tol=tol,
            random_state=random_state,
            verbose=verbose,
            early_stopping=early_stopping,
        )
        self.tau = tau
        self.regularisation = regularisation
        self.rank = rank
        self.positive = positive

    def _check_params(self):
        """check number of representations=2"""
        if self.n_views_ != 2:
            raise ValueError("SCCA_Span requires only 2 representations")
        self.max_obj = 0
        if self.regularisation == "l0":
            self.update = support_threshold
        elif self.regularisation == "l1":
            self.update = _delta_search
        self.tau = _process_parameter("tau", self.tau, 1, self.n_views_)
        self.positive = _process_parameter(
            "positive", self.positive, False, self.n_views_
        )

    def _update_weights(self, views: np.ndarray, i: int) -> None:
        """Update the weights_ for the i-th component.

        Args:
            views (np.ndarray): The input representations as numpy arrays.
            i (int): The index of the component.
        """
        # if P, D, Q not initialised, initialise them
        if getattr(self, "P", None) is None:
            self._initialize_variables(views)
        if i == 0:
            # generate a random vector c
            c = self.random_state.randn(self.rank)
            c /= np.linalg.norm(c)
            # compute a = P D c
            a = self.P @ np.diag(self.D) @ c
            # apply the update function to a with tau[0]
            u = self.update(a, self.tau[0])
            u /= np.linalg.norm(u)
            # update the objective value and the weights_ if improved
            return u[:, np.newaxis]
        elif i == 1:
            b = self.Q @ np.diag(self.D) @ self.P.T @ self.weights_[0]
            v = self.update(b, self.tau[1])
            v /= np.linalg.norm(v)
            return v

    def _initialize_variables(self, views):
        self.max_obj = [0, 0]
        cov = cross_cov(views[0], views[1], rowvar=False)
        # Perform SVD on im and obtain individual matrices
        P, D, Q = np.linalg.svd(cov, full_matrices=True)
        self.P = P[:, : self.rank]
        self.D = D[: self.rank]
        self.Q = Q[: self.rank, :].T
