from copy import copy

import numpy as np

from cca_zoo.models._stochastic._base import _BaseStochastic
from cca_zoo.utils import _process_parameter


class RCCAEigenGame(_BaseStochastic):
    """
    A class used to fit Regularized CCA by Delta-EigenGame

    Parameters
    ----------
    latent_dims : int, optional
        Number of latent dimensions to use, by default 1
    scale : bool, optional
        Whether to scale the data, by default True
    centre : bool, optional
        Whether to centre the data, by default True
    copy_data : bool, optional
        Whether to copy the data, by default True
    random_state : int, optional
        Random state to use, by default None
    accept_sparse : bool, optional
        Whether to accept sparse data, by default None
    batch_size : int, optional
        Batch size to use, by default 1
    shuffle : bool, optional
        Whether to shuffle the data, by default True
    sampler : torch.utils.data.Sampler, optional
        Sampler to use, by default None
    batch_sampler : torch.utils.data.Sampler, optional
        Batch sampler to use, by default None
    num_workers : int, optional
        Number of workers to use, by default 0
    pin_memory : bool, optional
        Whether to pin memory, by default False
    drop_last : bool, optional
        Whether to drop the last batch, by default True
    timeout : int, optional
        Timeout to use, by default 0
    worker_init_fn : function, optional
        Worker init function to use, by default None
    epochs : int, optional
        Number of epochs to use, by default 1
    learning_rate : float, optional
        Learning rate to use, by default 0.01
    c : float, optional
        Regularization parameter, by default 0

    References
    ----------
    Chapman, James, Ana Lawry Aguila, and Lennie Wells. "A Generalized EigenGame with Extensions to Multiview Representation Learning." arXiv preprint arXiv:2211.11323 (2022).
    """

    def __init__(
        self,
        latent_dims: int = 1,
        scale: bool = True,
        centre=True,
        copy_data=True,
        random_state=None,
        tol=1e-9,
        accept_sparse=None,
        batch_size=None,
        shuffle=True,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        timeout=0,
        worker_init_fn=None,
        epochs=1,
        learning_rate=1,
        c=0,
        nesterov=True,
        rho=0.1,
        line_search=False,
    ):
        super().__init__(
            latent_dims=latent_dims,
            scale=scale,
            centre=centre,
            copy_data=copy_data,
            accept_sparse=accept_sparse,
            random_state=random_state,
            tol=tol,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            epochs=epochs,
            learning_rate=learning_rate,
            nesterov=nesterov,
        )
        self.c = c
        self.rho = rho
        self.line_search = line_search

    def _check_params(self):
        self.c = _process_parameter("c", self.c, 0, self.n_views)

    def _update(self, views):
        converged = True
        if self.nesterov:
            v = self.weights
            for i in range(self.n_views):
                v[i] = self.weights[i] + self.momentum * (
                    self.weights[i] - self.weights_old[i]
                )
                self.weights_old[i] = self.weights[i].copy()
        else:
            v = self.weights
        for i, view in enumerate(views):
            projections = np.ma.stack([view @ weight for view, weight in zip(views, v)])
            Aw, Bw, wAw, wBw = self._get_terms(i, view, projections, v[i])
            grads = self.grads(Aw, wAw, Bw, wBw)
            if self.line_search:
                w_ = self._backtracking_line_search(i, views, v, grads)
            else:
                w_ = v[i] - self.learning_rate * grads
            # if difference between self.weights[i] and w_ is less than tol, then return True
            if not np.allclose(self.weights[i], w_[i], atol=self.tol):
                converged = False
            self.weights[i] = w_
            v[i] = self.weights[i]
        return converged

    def grads(self, Aw, wAw, Bw, wBw):
        return -2 * Aw + (Aw @ np.triu(wBw) + Bw @ np.triu(wAw))

    def _Aw(self, view, projections):
        return view.T @ projections / view.shape[0]

    def _Bw(self, view, projection, weight, c):
        if c==1:
            return (c * weight)
        else:
            return (c * weight) + (1 - c) * (view.T @ projection) / projection.shape[0]

    def _get_terms(self, i, view, projections, v):
        projections.mask[i] = True
        Aw = self._Aw(view, projections.sum(axis=0).filled())
        projections.mask[i] = False
        Bw = self._Bw(view, projections[i].filled(), v, self.c[i])
        wAw = v.T @ Aw
        wBw = v.T @ Bw
        wAw[np.diag_indices_from(wAw)] = np.where(np.diag(wAw) > 0, np.diag(wAw), 0)
        wBw[np.diag_indices_from(wBw)] = np.where(np.diag(wAw) > 0, np.diag(wBw), 0)
        return Aw, Bw, wAw, wBw

    def objective(self, views, weights, k=None):
        if k is None:
            k = self.latent_dims
        projections = np.ma.stack([view @ weight for view, weight in zip(views, weights)])
        objective = 0
        for i, view in enumerate(views):
            Aw, Bw, wAw, wBw = self._get_terms(i, view, projections, weights[i])
            objective -= 2 * np.trace(wAw[:k+1,:k+1]) - np.trace(wAw[:k+1,:k+1] @ wBw[:k+1,:k+1])
        return objective

    def _backtracking_line_search(self, i,views, v, grad):
        t = [self.learning_rate] * grad.shape[1]
        w_new = [v_.copy() for v_ in v]
        for k in range(grad.shape[1]):
            f = self.objective(views, w_new, k=k)
            while True:
                # Compute the candidate weight vector using the proximal operator
                w_new[i][:, k] = v[i][:, k] - t[k] * grad[:, k]
                # Compute the candidate objective function value
                f_new = self.objective(views, w_new, k=k)
                # Check the sufficient decrease condition
                if (f_new <= f + t[k] * grad[:, k].T @ (w_new[i][:, k] - v[i][:, k]))or (t[k] < 1e-9):
                    break
                t[k] *= self.rho
        return w_new[i]


class CCAEigenGame(RCCAEigenGame):
    """
    A class used to fit CCA by Delta-EigenGame

    Parameters
    ----------
    latent_dims : int, optional
        Number of latent dimensions to use, by default 1
    scale : bool, optional
        Whether to scale the data, by default True
    centre : bool, optional
        Whether to centre the data, by default True
    copy_data : bool, optional
        Whether to copy the data, by default True
    random_state : int, optional
        Random state to use, by default None
    accept_sparse : bool, optional
        Whether to accept sparse data, by default None
    batch_size : int, optional
        Batch size to use, by default 1
    shuffle : bool, optional
        Whether to shuffle the data, by default True
    sampler : torch.utils.data.Sampler, optional
        Sampler to use, by default None
    batch_sampler : torch.utils.data.Sampler, optional
        Batch sampler to use, by default None
    num_workers : int, optional
        Number of workers to use, by default 0
    pin_memory : bool, optional
        Whether to pin memory, by default False
    drop_last : bool, optional
        Whether to drop the last batch, by default True
    timeout : int, optional
        Timeout to use, by default 0
    worker_init_fn : function, optional
        Worker init function to use, by default None
    epochs : int, optional
        Number of epochs to use, by default 1
    learning_rate : float, optional
        Learning rate to use, by default 0.01

    References
    ----------
    Chapman, James, Ana Lawry Aguila, and Lennie Wells. "A Generalized EigenGame with Extensions to Multiview Representation Learning." arXiv preprint arXiv:2211.11323 (2022).
    """

    def __init__(
        self,
        latent_dims: int = 1,
        scale: bool = True,
        centre=True,
        copy_data=True,
        random_state=None,
        tol=1e-9,
        accept_sparse=None,
        batch_size=None,
        shuffle=True,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        timeout=0,
        worker_init_fn=None,
        epochs=1,
        learning_rate=1,
        nesterov=True,
        line_search=False,
    ):
        super().__init__(
            latent_dims=latent_dims,
            scale=scale,
            centre=centre,
            copy_data=copy_data,
            accept_sparse=accept_sparse,
            random_state=random_state,
            tol=tol,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            epochs=epochs,
            learning_rate=learning_rate,
            c=0,
            nesterov=nesterov,
            line_search=line_search,
        )


class PLSEigenGame(RCCAEigenGame):
    """
    A class used to fit PLS by Delta-EigenGame

    Parameters
    ----------
    latent_dims : int, optional
        Number of latent dimensions to use, by default 1
    scale : bool, optional
        Whether to scale the data, by default True
    centre : bool, optional
        Whether to centre the data, by default True
    copy_data : bool, optional
        Whether to copy the data, by default True
    random_state : int, optional
        Random state to use, by default None
    accept_sparse : bool, optional
        Whether to accept sparse data, by default None
    batch_size : int, optional
        Batch size to use, by default 1
    shuffle : bool, optional
        Whether to shuffle the data, by default True
    sampler : torch.utils.data.Sampler, optional
        Sampler to use, by default None
    batch_sampler : torch.utils.data.Sampler, optional
        Batch sampler to use, by default None
    num_workers : int, optional
        Number of workers to use, by default 0
    pin_memory : bool, optional
        Whether to pin memory, by default False
    drop_last : bool, optional
        Whether to drop the last batch, by default True
    timeout : int, optional
        Timeout to use, by default 0
    worker_init_fn : function, optional
        Worker init function to use, by default None
    epochs : int, optional
        Number of epochs to use, by default 1
    learning_rate : float, optional
        Learning rate to use, by default 0.01

    References
    ----------
    Chapman, James, Ana Lawry Aguila, and Lennie Wells. "A Generalized EigenGame with Extensions to Multiview Representation Learning." arXiv preprint arXiv:2211.11323 (2022).
    """

    def __init__(
        self,
        latent_dims: int = 1,
        scale: bool = True,
        centre=True,
        copy_data=True,
        random_state=None,
        tol=1e-9,
        accept_sparse=None,
        batch_size=None,
        shuffle=True,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        timeout=0,
        worker_init_fn=None,
        epochs=1,
        learning_rate=1,
        nesterov=True,
        line_search=False,
    ):
        super().__init__(
            latent_dims=latent_dims,
            scale=scale,
            centre=centre,
            copy_data=copy_data,
            accept_sparse=accept_sparse,
            random_state=random_state,
            tol=tol,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            epochs=epochs,
            learning_rate=learning_rate,
            c=1,
            nesterov=nesterov,
            line_search=line_search,
        )
