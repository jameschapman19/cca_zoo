from typing import Union

import numpy as np

from cca_zoo.models._iterative._base import _BaseInnerLoop
from ._altmaxvar import AltMaxVar


class PDD_GCCA(AltMaxVar):
    r"""
    Fits a Primal Dual Decomposition Regularized CCA model to two or more views of data.

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{\sum_i\sum_{j\neq i} \|X_iw_i-X_jw_j\|^2 + c\|w_i\|^2_2 + \text{l1_ratio}\|w_i\|_1\}\\

        \text{subject to:}

        w_i^TX_i^TX_iw_i=n

    References
    ----------
    Kanatsoulis, Charilaos I., et al. "Structured SUMCOR multiview canonical correlation analysis for large-scale data." IEEE Transactions on Signal Processing 67.2 (2018): 306-319.

    """

    def __init__(
        self,
        latent_dims: int = 1,
        scale: bool = True,
        centre=True,
        copy_data=True,
        random_state=None,
        max_iter: int = 100,
        initialization: Union[str, callable] = "pls",
        tol: float = 1e-9,
        view_regs=None,
        verbose=0,
    ):
        super().__init__(
            latent_dims=latent_dims,
            scale=scale,
            centre=centre,
            copy_data=copy_data,
            max_iter=max_iter,
            initialization=initialization,
            tol=tol,
            random_state=random_state,
            verbose=verbose,
        )
        self.view_regs = view_regs

    def _set_loop_params(self):
        self.loop = _PDD_GCCALoop(
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            view_regs=self.view_regs,
            verbose=self.verbose,
        )


class _PDD_GCCALoop(_BaseInnerLoop):
    def __init__(
        self,
        max_iter: int = 100,
        tol=1e-9,
        random_state=None,
        view_regs=None,
        alpha=1e-3,
        eta=1e-3,
        rho=1e-3,
        c=0.9,
        eps=1e-3,
        verbose=0,
    ):
        super().__init__(
            max_iter=max_iter, tol=tol, random_state=random_state, verbose=verbose
        )
        self.alpha = alpha
        self.view_regs = view_regs
        self.eta = eta
        self.rho = rho
        self.c = c
        self.eps = eps

    def _inner_iteration(self, views):
        # Update each view using loop update function
        self._update_Y()
        for i, view in enumerate(views):
            # if no nans
            if np.isnan(self.scores).sum() == 0:
                self._update_view(views, i)

    def _update_Y(self):
        if np.linalg.norm(self.scores - self.G, axis=0).sum() < self.eta:
            self.Y = self.Y + self.rho * (self.scores - self.G)
        else:
            self.rho = self.rho * self.c

    def _update_view(self, views, view_index: int):
        converged = False
        while not converged:
            targets = np.ma.array(self.scores, mask=False)
            targets.mask[view_index] = True
            target = (
                targets.sum(axis=0).filled()
                + self.G[view_index]
                - self.Y[view_index] / self.rho
            )
            weights_ = self.view_regs[view_index](
                (self.n_views + self.rho) * views[view_index],
                target,
                self.weights[view_index],
            )
            U, _, Vt = np.linalg.svd(
                targets.sum(axis=0).filled()
                + self.rho * self.scores[view_index]
                + self.Y[view_index],
                full_matrices=False,
            )
            G_ = U @ Vt
            if (
                max(
                    np.linalg.norm(weights_ - self.weights[view_index], ord=np.inf),
                    np.linalg.norm(G_ - self.G[view_index], ord=np.inf),
                )
                < self.eps
            ):
                converged = True
            self.weights[view_index] = weights_
            self.G[view_index] = G_
        self.scores[view_index] = views[view_index] @ self.weights[view_index]

    def _objective(self, views):
        total_objective = 0
        for i, _ in enumerate(views):
            objective = (
                np.linalg.norm(
                    views[i] @ self.weights[i] - self.scores, ord="fro", axis=(1, 2)
                )
                ** 2
            ).sum() / 2
            total_objective += objective + self.view_regs[i].cost(
                views[i], self.weights[i]
            )
        return total_objective

    def _early_stop(self) -> bool:
        # Some kind of early stopping
        if np.abs(self.track["objective"][-2] - self.track["objective"][-1]) < 1e-9:
            return True
        else:
            return False

    def _initialize(self, views):
        self.weights = [
            np.zeros((view.shape[1], self.scores[0].shape[1])) for view in views
        ]
        self.G = self.scores.copy()
        self.Y = np.zeros_like(self.G)
