from typing import Union, Iterable

import numpy as np

from ._base import _BaseIterative
from ...utils import _process_parameter, _check_converged_weights, _check_Parikh2014


class SCCA_ADMM(_BaseIterative):
    r"""
    Fits a sparse CCA model by alternating ADMM for two or more views.

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{\sum_i\sum_{j\neq i} \|X_iw_i-X_jw_j\|^2 + \|w_i\|_1\}\\

        \text{subject to:}

        w_i^TX_i^TX_iw_i=1

    Parameters
    ----------
    latent_dims : int, default=1
        Number of latent dimensions to use in the model.
    scale : bool, default=True
        Whether to scale the data to unit variance.
    centre : bool, default=True
        Whether to centre the data to have zero mean.
    copy_data : bool, default=True
        Whether to copy the data or overwrite it.
    random_state : int, default=None
        Random seed for initialisation.
    deflation : str, default="cca"
        Deflation method to use. Options are "cca" and "pls".
    tau : float or list of floats, default=None
        Regularisation parameter. If a single float is given, the same value is used for all views.
        If a list of floats is given, the values are used for each view.
    mu : float or list of floats, default=None
        Regularisation parameter. If a single float is given, the same value is used for all views.
        If a list of floats is given, the values are used for each view.
    lam : float or list of floats, default=None
        Regularisation parameter. If a single float is given, the same value is used for all views.
        If a list of floats is given, the values are used for each view.
    eta : float or list of floats, default=None
        Regularisation parameter. If a single float is given, the same value is used for all views.
        If a list of floats is given, the values are used for each view.
    max_iter : int, default=100
        Maximum number of iterations to run.
    initialization : str or callable, default="pls"
        Method to use for initialisation. Options are "pls" and "random".
    tol : float, default=1e-9
        Tolerance for convergence.
    verbose : int, default=0
        Verbosity level. If 0, no output is printed. If 1, output is printed every 10 iterations.



    References
    ----------
    Suo, Xiaotong, et al. "Sparse canonical correlation analysis." arXiv preprint arXiv:1705.10865 (2017).

    Examples
    --------
    >>> from cca_zoo.models import SCCA_ADMM
    >>> import numpy as np
    >>> rng=np.random.RandomState(0)
    >>> X1 = rng.random((10,5))
    >>> X2 = rng.random((10,5))
    >>> model = SCCA_ADMM(random_state=0,tau=[1e-1,1e-1])
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
        tau: Union[Iterable[float], float] = None,
        mu: Union[Iterable[float], float] = None,
        lam: Union[Iterable[float], float] = None,
        eta: Union[Iterable[float], float] = None,
        max_iter: int = 100,
        initialization: Union[str, callable] = "pls",
        tol: float = 1e-3,
        verbose=0,
    ):
        self.tau = tau
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

    def _check_params(self):
        self.tau = _process_parameter("tau", self.tau, 0, self.n_views)
        self.lam = _process_parameter("lam", self.lam, 1, self.n_views)
        self.eta = _process_parameter("eta", self.eta, 0, self.n_views)

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

    def _update(self, views, scores, weights):
        # Update each view using loop update function
        for view_index, view in enumerate(views):
            targets = np.ma.array(scores, mask=False)
            targets.mask[view_index] = True
            gradient = views[view_index].T @ targets.sum(axis=0).filled()
            mu = self.mu[view_index]
            lam = self.lam[view_index]
            N = views[view_index].shape[0]
            unnorm_z = []
            norm_eta = []
            norm_weights = []
            norm_proj = []
            for _ in range(self.max_iter):
                # We multiply 'tau' by N in order to make regularisation match across the different sparse cca methods
                weights[view_index] = self._prox_mu_f(
                    weights[view_index]
                    - mu
                    / lam
                    * views[view_index].T
                    @ (
                        views[view_index] @ weights[view_index]
                        - self.z[view_index]
                        + self.eta[view_index]
                    ),
                    mu,
                    gradient,
                    N * self.tau[view_index],
                )
                unnorm_z.append(
                    np.linalg.norm(
                        views[view_index] @ weights[view_index] + self.eta[view_index]
                    )
                )
                self.z[view_index] = self._prox_lam_g(
                    views[view_index] @ weights[view_index] + self.eta[view_index]
                )
                self.eta[view_index] = (
                    self.eta[view_index]
                    + views[view_index] @ weights[view_index]
                    - self.z[view_index]
                )
                norm_eta.append(np.linalg.norm(self.eta[view_index]))
                norm_proj.append(
                    np.linalg.norm(views[view_index] @ weights[view_index])
                )
                norm_weights.append(np.linalg.norm(weights[view_index], 1))
            _check_converged_weights(weights[view_index], view_index)
            scores[view_index] = views[view_index] @ weights[view_index]
        return scores, weights

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
