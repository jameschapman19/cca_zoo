from typing import Union

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
        initialization: Union[str, callable] = "random",
        c=0,
        nesterov=True,
        line_search=False,
        rho=0.1,
        ensure_descent=False,
        component_wise=False,
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
            initialization=initialization,
            nesterov=nesterov,
        )
        self.c = c
        self.rho = rho
        self.line_search = line_search
        self.ensure_descent = ensure_descent
        self.component_wise = component_wise
        # if line search and not component wise, warn user
        if self.line_search and not self.component_wise:
            print(
                "Warning: Line search is not recommended when not using component-wise updates"
            )
        self.y = None
        self.u = None

    def _check_params(self):
        self.c = _process_parameter("c", self.c, 0, self.n_views)

    def _split_weights(self, views, weights):
        splits = np.cumsum([0] + [view.shape[1] for view in views])
        weights = [
            weights[split : splits[i + 1]] for i, split in enumerate(splits[:-1])
        ]
        return weights

    def _update(self, views):
        # combine weights
        self.weights = np.vstack(self.weights)
        if self.component_wise:
            for k in range(self.weights.shape[1]):
                self._update_component(views, k=k)
        else:
            self._update_all(views)
        # split weights
        self.weights = self._split_weights(views, self.weights)

    def _update_all(self, views):
        if self.y is None or self.u is None:
            self.y = self.u = self.weights.copy()
        # nesterov momentum
        if self.nesterov:
            if self.u is None:
                self.u = self.weights.copy()
            self.y = self.weights + self.momentum * (self.u - self.weights)
        else:
            self.y = self.weights.copy()
        grads = self.grads(views, u=self.y)
        if self.line_search:
            self.learning_rate = self._backtracking_line_search(views)
        t = self.learning_rate
        self.u = self._gradient_step(self.y, t, grads)
        if self.ensure_descent:
            if self.objective(views, u=self.u) < self.objective(views, u=self.weights):
                self.weights = self.u.copy()
        else:
            self.weights = self.u.copy()

    def _update_component(self, views, k=0):
        if self.y is None or self.u is None:
            self.y = self.u = self.weights.copy()
        # nesterov momentum
        if self.nesterov:
            if self.u is None:
                self.u = self.weights.copy()
            self.y[:, k] = self.weights[:, k] + self.momentum * (
                self.u[:, k] - self.weights[:, k]
            )
        else:
            self.y[:, k] = self.weights[:, k].copy()
        grads = self.grads(views, u=self.y)
        if self.line_search:
            t=self._backtracking_line_search(views, k=k)
        else:
            t = self.learning_rate
        self.u[:, k] = self._gradient_step(self.y[:, k], t, grads[:, k])
        if self.ensure_descent:
            if self.objective(views, u=self.u, k=k) < self.objective(
                views, u=self.weights, k=k
            ):
                self.weights[:, k] = self.u[:, k].copy()
        else:
            self.weights[:, k] = self.u[:, k].copy()

    def _backtracking_line_search(self, views, k=None, beta=0.8):
        # Initialize the gradient vector
        if k is None:
            gradient = self.grads(views, u=self.y)
        else:
            gradient = self.grads(views, u=self.y)[:, k]

        # Initialize the step size
        step_size = self.learning_rate

        # Initialize the updated solution vector
        updated_solution = self.y.copy()

        # Perform backtracking line search until the objective function decreases
        while True:
            # Update the solution vector based on the gradient and step size
            if k is None:
                updated_solution = self._gradient_step(self.y, step_size, gradient)
            else:
                updated_solution[:, k] = self._gradient_step(
                    self.y[:, k], step_size, gradient
                )

            # Check if the objective function decreases with the updated solution
            if (
                self.objective(views, u=updated_solution, k=k)
                < self.objective(views, u=self.y, k=k)
                - 0.5 * step_size * np.linalg.norm(gradient) ** 2
            ):
                break

            # Reduce the step size by a factor of beta
            step_size *= beta

            # Set a lower bound for the step size to avoid numerical issues
            if step_size < 1e-6:
                step_size = 1e-6
                break

        return step_size

    def grads(self, views, u=None):
        Aw, Bw, wAw, wBw = self._get_terms(views, u)
        grads = 2 * Aw - (Aw @ np.triu(wBw) * np.sign(np.diag(wAw)) + Bw @ np.triu(wAw))
        # TODO: work out why this works
        # =2 * Aw - (Aw @ np.triu(wBw) + Bw @ np.triu(wAw)) for some reason multiplying by the sign fixes the negative eigenvalue problem here. Not sure why this is the case.
        return -grads

    def _Aw(self, views, projections):
        Aw = np.vstack(
            [
                view.T @ projections.sum(axis=0) / projections[0].shape[0]
                for view in views
            ]
        )
        return Aw / len(views)

    def _Bw(self, views, projections, u):
        weights = self._split_weights(views, u)
        Bw = []
        for i, (view, projection, weight, c) in enumerate(
            zip(views, projections, weights, self.c)
        ):
            if c == 1:
                Bw.append(c * weight)
            else:
                Bw.append(
                    (c * weight) + (1 - c) * (view.T @ projection) / projection.shape[0]
                )
        return np.vstack(Bw) / len(views)

    def _get_terms(self, views, u, projections=None):
        if projections is None:
            projections = self.projections(views, u)
        Aw = self._Aw(views, projections)
        Bw = self._Bw(views, projections, u)
        wAw = u.T @ Aw
        wBw = u.T @ Bw
        return Aw, Bw, wAw, wBw

    def objective(self, views, u=None, k=None, projections=None):
        # if u is a list type
        if isinstance(u, list):
            u = np.vstack(u)
        if k is None:
            k = self.latent_dims
        if projections is None:
            projections = self.projections(views, u)
        Aw, Bw, wAw, wBw = self._get_terms(views, u, projections=projections)
        return -2 * np.trace(wAw[: k + 1, : k + 1]) + np.trace(
            wAw[: k + 1, : k + 1] @ wBw[: k + 1, : k + 1]
        )

    def projections(self, views, u):
        u = self._split_weights(views, u)
        m = []
        for view, weight in zip(views, u):
            n = view @ weight
            m.append(n)
        return np.stack(m)

    def _gradient_step(self, y, lr, grad):
        return y - lr * grad

    def _more_tags(self):
        return {"multiview": True, "stochastic": True}


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
        rho=0.1,
        ensure_descent=False,
        component_wise=False,
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
            rho=rho,
            ensure_descent=ensure_descent,
            component_wise=component_wise,
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
        rho=0.1,
        ensure_descent=False,
        component_wise=False,
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
            rho=rho,
            ensure_descent=ensure_descent,
            component_wise=component_wise,
        )

    def _Aw(self, views, projections):
        Aw = np.vstack(
            [
                view.T @ projections.sum(axis=0) / projections[0].shape[0]
                for view in views
            ]
        ) - np.vstack(
            [
                view.T @ projection / projections[0].shape[0]
                for view, projection in zip(views, projections)
            ]
        )
        return Aw / len(views)
