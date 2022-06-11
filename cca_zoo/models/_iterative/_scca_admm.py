from typing import Union, Iterable

import numpy as np

from ._base import _BaseIterative
from ._elastic import _ElasticInnerLoop
from ...utils import _process_parameter, _check_converged_weights, _check_Parikh2014


class SCCA_ADMM(_BaseIterative):
    r"""
    Fits a sparse CCA model by alternating ADMM

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{\sum_i\sum_{j\neq i} \|X_iw_i-X_jw_j\|^2 + \|w_i\|_1\}\\

        \text{subject to:}

        w_i^TX_i^TX_iw_i=1

    :Citation:

    Suo, Xiaotong, et al. "Sparse canonical correlation analysis." arXiv preprint arXiv:1705.10865 (2017).

    :Example:

    >>> from cca_zoo.models import SCCA_ADMM
    >>> import numpy as np
    >>> rng=np.random.RandomState(0)
    >>> X1 = rng.random((10,5))
    >>> X2 = rng.random((10,5))
    >>> model = SCCA_ADMM(random_state=0,c=[1e-1,1e-1])
    >>> model.fit((X1,X2)).score((X1,X2))
    array([0.84348183])
    """

    def __init__(
        self,
        latent_dims: int = 1,
        scale: bool = True,
        centre=True,
        copy_data=True,
        random_state=None,
        deflation="cca",
        c: Union[Iterable[float], float] = None,
        mu: Union[Iterable[float], float] = None,
        lam: Union[Iterable[float], float] = None,
        eta: Union[Iterable[float], float] = None,
        max_iter: int = 100,
        initialization: Union[str, callable] = "pls",
        tol: float = 1e-9,
        verbose=0,
    ):
        """
        Constructor for SCCA_ADMM

        :param latent_dims: number of latent dimensions to fit
        :param scale: normalize variance in each column before fitting
        :param centre: demean data by column before fitting (and before transforming out of sample
        :param copy_data: If True, views will be copied; else, it may be overwritten
        :param random_state: Pass for reproducible output across multiple function calls
        :param c: l1 regularisation parameter
        :param max_iter: the maximum number of iterations to perform in the inner optimization loop
        :param initialization: either string from "pls", "cca", "random", "uniform" or callable to initialize the score variables for _iterative methods
        :param tol: tolerance value used for early stopping
        :param mu:
        :param lam:
        :param: eta:
        """
        self.c = c
        self.mu = mu
        self.lam = lam
        self.eta = eta
        super().__init__(
            latent_dims=latent_dims,
            scale=scale,
            centre=centre,
            copy_data=copy_data,
            max_iter=max_iter,
            initialization=initialization,
            tol=tol,
            random_state=random_state,
            deflation=deflation,
            verbose=verbose,
        )

    def _set_loop_params(self):
        self.loop = _ADMMInnerLoop(
            max_iter=self.max_iter,
            c=self.c,
            mu=self.mu,
            lam=self.lam,
            eta=self.eta,
            tol=self.tol,
            random_state=self.random_state,
            verbose=self.verbose,
        )

    def _check_params(self):
        self.c = _process_parameter("c", self.c, 0, self.n_views)
        self.lam = _process_parameter("lam", self.lam, 1, self.n_views)
        self.eta = _process_parameter("eta", self.eta, 0, self.n_views)


class _ADMMInnerLoop(_ElasticInnerLoop):
    def __init__(
        self,
        max_iter: int = 100,
        tol=1e-9,
        mu=None,
        lam=None,
        c=None,
        eta=None,
        random_state=None,
        verbose=0,
    ):
        super().__init__(
            max_iter=max_iter, tol=tol, random_state=random_state, verbose=verbose
        )
        self.c = c
        self.lam = lam
        self.mu = mu
        self.eta = eta

    def _initialize(self, views):
        self.eta = [np.ones(view.shape[0]) * eta for view, eta in zip(views, self.eta)]
        self.z = [np.zeros(view.shape[0]) for view in views]
        if self.mu is None:
            self.mu = [
                lam / np.linalg.norm(view) ** 2 for lam, view in zip(self.lam, views)
            ]
        else:
            self.mu = _process_parameter("mu", self.mu, 0, self.n_views)
        _check_Parikh2014(self.mu, self.lam, views)
        self.l1_ratio = _process_parameter("c", self.c, 1, self.n_views)

    def _update_view(self, views, view_index: int):
        targets = np.ma.array(self.scores, mask=False)
        targets.mask[view_index] = True
        # Suo uses parameter tau whereas we use parameter c to penalize the 1-norm of the weights.
        # Suo uses c to refer to the gradient where we now use gradient
        gradient = views[view_index].T @ targets.sum(axis=0).filled()
        # reset eta each loop?
        # self.eta[view_index][:] = 0
        mu = self.mu[view_index]
        lam = self.lam[view_index]
        N = views[view_index].shape[0]
        unnorm_z = []
        norm_eta = []
        norm_weights = []
        norm_proj = []
        for _ in range(self.max_iter):
            # We multiply 'c' by N in order to make regularisation match across the different sparse cca methods
            self.weights[view_index] = self._prox_mu_f(
                self.weights[view_index]
                - mu
                / lam
                * views[view_index].T
                @ (
                    views[view_index] @ self.weights[view_index]
                    - self.z[view_index]
                    + self.eta[view_index]
                ),
                mu,
                gradient,
                N * self.c[view_index],
            )
            unnorm_z.append(
                np.linalg.norm(
                    views[view_index] @ self.weights[view_index] + self.eta[view_index]
                )
            )
            self.z[view_index] = self._prox_lam_g(
                views[view_index] @ self.weights[view_index] + self.eta[view_index]
            )
            self.eta[view_index] = (
                self.eta[view_index]
                + views[view_index] @ self.weights[view_index]
                - self.z[view_index]
            )
            norm_eta.append(np.linalg.norm(self.eta[view_index]))
            norm_proj.append(
                np.linalg.norm(views[view_index] @ self.weights[view_index])
            )
            norm_weights.append(np.linalg.norm(self.weights[view_index], 1))
        _check_converged_weights(self.weights[view_index], view_index)
        self.scores[view_index] = views[view_index] @ self.weights[view_index]

    def _prox_mu_f(self, x, mu, c, tau):
        u_update = x.copy()
        mask_1 = x + (mu * c) > mu * tau
        # if mask_1.sum()>0:
        u_update[mask_1] = x[mask_1] + mu * (c[mask_1] - tau)
        mask_2 = x + (mu * c) < -mu * tau
        # if mask_2.sum() > 0:
        u_update[mask_2] = x[mask_2] + mu * (c[mask_2] + tau)
        mask_3 = ~(mask_1 | mask_2)
        u_update[mask_3] = 0
        return u_update

    def _prox_lam_g(self, x):
        norm = np.linalg.norm(x)
        if norm < 1:
            return x
        else:
            return x / max(1, norm)
