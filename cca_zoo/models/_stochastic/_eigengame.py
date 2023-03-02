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
        learning_rate=1e-3,
        c=0,
        nesterov=True,
        rho=0.1,
        line_search=False,
        ensure_descent=True,
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
        self.ensure_descent = ensure_descent

    def _check_params(self):
        self.c = _process_parameter("c", self.c, 0, self.n_views)

    def _update(self, views):
        for i, view in enumerate(views):
            for k in range(self.latent_dims):
                y = [w.copy() for w in self.weights]
                y[i][:, k] = self.weights[i][:, k] + self.momentum * (self.u[i] - self.weights_old[i])[:, k]
                if self.line_search:
                    u = self._backtracking_line_search(i, views, y, k)
                else:
                    u = self._nesterov_gradient_descent(i, views, y, k)
                self.u[i][:, k] = u[i][:, k]
                self.weights_old[i][:, k] = self.weights[i][:, k].copy()
                if self.ensure_descent:
                    # ensure descent http://www.seas.ucla.edu/~vandenbe/236C/lectures/fista.pdf
                    if self.objective(views, self.u, k=k) <= self.objective(views, self.weights, k=k):
                        self.weights[i][:, k] = self.u[i][:, k]
        return False

    def grads(self, Aw, wAw, Bw, wBw):
        return -2 * Aw + (Aw @ np.triu(wBw) + Bw @ np.triu(wAw))

    def _Aw(self, view, projections):
        return view.T @ projections / view.shape[0]

    def _Bw(self, view, projection, weight, c):
        if c == 1:
            return c * weight
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

    def objective(self, views, weights, k=None, projections=None):
        if k is None:
            k = self.latent_dims
        if projections is None:
            projections = self.projections(views, weights)
        objective = 0
        for i, view in enumerate(views):
            Aw, Bw, wAw, wBw = self._get_terms(i, view, projections, weights[i])
            objective -= 2 * np.trace(wAw[: k + 1, : k + 1]) - np.trace(
                wAw[: k + 1, : k + 1] @ wBw[: k + 1, : k + 1]
            )
        return objective

    def projections(self, views, weights):
        m = []
        for view, weight in zip(views, weights):
            n = view @ weight
            m.append(n)
        return np.ma.stack(m)

    def _backtracking_line_search(self, i, views, y, k):
        projections = np.ma.stack([view @ weight for view, weight in zip(views, y)])
        u = [y_.copy() for y_ in y]
        t = 1
        f = self.objective(views, u, k=k, projections=projections)
        Aw, Bw, wAw, wBw = self._get_terms(i, views[i], projections, u[i])
        grad = self.grads(Aw, wAw, Bw, wBw)
        while True:
            # Compute the candidate weight vector using the proximal operator
            u[i][:, k] = y[i][:, k] - t * grad[:, k]
            projections[i][:, k] = views[i] @ u[i][:, k]
            # Compute the candidate objective function value
            f_new = self.objective(views, u, k=k, projections=projections)
            # Check the sufficient decrease condition
            if (
                    f_new
                    <= f
                    + t * grad[:, k] @ (u[i][:, k] - y[i][:, k])
                    + 0.5 * t * np.linalg.norm(u[i][:, k] - y[i][:, k]) ** 2
            ) or (t < self.learning_rate):
                break
            t *= self.rho
            if t < self.learning_rate:
                u[i][:, k] = y[i][:, k]
                break
        return u

    def _nesterov_gradient_descent(self, i, views, y, k):
        projections = np.ma.stack([view @ weight for view, weight in zip(views, y)])
        u = [y_.copy() for y_ in y]
        Aw, Bw, wAw, wBw = self._get_terms(i, views[i], projections, u[i])
        grad = self.grads(Aw, wAw, Bw, wBw)
        u[i][:, k] = y[i][:, k] - self.learning_rate * grad[:, k]
        return u


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
        learning_rate=1e-3,
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
        learning_rate=1e-3,
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
