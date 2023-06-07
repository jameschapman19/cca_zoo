from typing import Iterable, Union

import numpy as np

from cca_zoo.models._iterative._base import BaseIterative, BaseLoop
from cca_zoo.models._search import _delta_search, support_threshold
from cca_zoo.utils import _process_parameter


class SCCA_Span(BaseIterative):
    r"""
    Fits a Sparse CCA model using SpanCCA.

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{\sum_i\sum_{j\neq i} \|X_iw_i-X_jw_j\|^2 + \text{l1_ratio}\|w_i\|_1\}\\

        \text{subject to:}

        w_i^TX_i^TX_iw_i=1

    References
    ----------
    Asteris, Megasthenis, et al. "A simple and provable algorithm for sparse diagonal CCA." International Conference on Machine Learning. PMLR, 2016.


    Examples
    --------
    >>> from cca_zoo.models import SCCA_Span
    >>> import numpy as np
    >>> rng=np.random.RandomState(0)
    >>> X1 = rng.random((10,5))
    >>> X2 = rng.random((10,5))
    >>> model = SCCA_Span(regularisation="l0", tau=[2, 2])
    >>> model.fit((X1,X2)).score((X1,X2))
    array([0.84556666])
    """

    def __init__(
        self,
        latent_dims: int = 1,
        epochs: int = 100,
        copy_data=True,
        initialization: str = "uniform",
        tol: float = 1e-3,
        regularisation="l0",
        tau: Union[Iterable[Union[float, int]], Union[float, int]] = None,
        rank=1,
        positive: Union[Iterable[bool], bool] = None,
        random_state=None,
        deflation="cca",
        verbose=0,
    ):
        super().__init__(
            latent_dims=latent_dims,
            epochs=epochs,
            copy_data=copy_data,
            initialization=initialization,
            tol=tol,
            random_state=random_state,
            deflation=deflation,
            verbose=verbose,
        )
        self.tau = tau
        self.regularisation = regularisation
        self.rank = rank
        self.positive = positive

    def _check_params(self):
        """check number of views=2"""
        if self.n_views_ != 2:
            raise ValueError(f"SCCA_Span requires only 2 views")
        self.max_obj = 0
        if self.regularisation == "l0":
            self.update = support_threshold
        elif self.regularisation == "l1":
            self.update = _delta_search
        self.tau = _process_parameter("tau", self.tau, 0, self.n_views_)
        self.positive = _process_parameter(
            "positive", self.positive, False, self.n_views_
        )

    def _get_module(self, weights=None, k=None):
        return SpanLoop(
            weights=weights,
            k=k,
            regularisation=self.regularisation,
            tau=self.tau,
            positive=self.positive,
            rank=self.rank,
            random_state=self.random_state,
        )


class SpanLoop(BaseLoop):
    def __init__(
        self,
        weights,
        k=None,
        regularisation="l0",
        tau=None,
        positive=False,
        rank=1,
        random_state=None,
    ):
        super().__init__(weights=weights, k=k)
        self.regularisation = regularisation
        self.tau = tau
        self.positive = positive
        if self.regularisation == "l0":
            self.update = support_threshold
        elif self.regularisation == "l1":
            self.update = _delta_search
        self.rank = rank
        self.random_state = random_state

    def training_step(self, batch, batch_idx):
        # if P, D, Q not initialised, initialise them
        if getattr(self, "P", None) is None:
            self._initialize(batch["views"])
        c = self.random_state.randn(self.rank)
        c /= np.linalg.norm(c)
        a = self.P @ np.diag(self.D) @ c
        u = self.update(a, self.tau[0])
        u /= np.linalg.norm(u)
        b = self.Q @ np.diag(self.D) @ self.P.T @ u
        v = self.update(b, self.tau[1])
        v /= np.linalg.norm(v)
        if b.T @ v > self.max_obj:
            self.max_obj = b.T @ v
            self.weights[0] = u
            self.weights[1] = v

    def _initialize(self, views):
        self.max_obj = 0
        cov = views[0].T @ views[1] / views[0].shape[0]
        # Perform SVD on im and obtain individual matrices
        P, D, Q = np.linalg.svd(cov, full_matrices=True)
        self.P = P[:, : self.rank]
        self.D = D[: self.rank]
        self.Q = Q[: self.rank, :].T
