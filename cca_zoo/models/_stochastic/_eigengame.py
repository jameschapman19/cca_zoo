from typing import Union

import numpy as np

from cca_zoo.models._stochastic._base import _BaseStochastic


class CCAEigenGame(_BaseStochastic):
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
        momentum=0,
        line_search=False,
        rho=0.1,
        ensure_descent=False,
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
            momentum=momentum,
        )
        self.rho = rho
        self.line_search = line_search
        self.ensure_descent = ensure_descent
        self.velocity = None

    def _split_weights(self, views, weights):
        splits = np.cumsum([0] + [view.shape[1] for view in views])
        weights = [
            weights[split : splits[i + 1]] for i, split in enumerate(splits[:-1])
        ]
        return weights

    def _update(self, views):
        # combine weights
        self.weights = np.vstack(self.weights)
        self._update_all(views)
        # split weights
        self.weights = self._split_weights(views, self.weights)

    def _update_all(self, views):
        if self.velocity is None:
            self.velocity = np.zeros_like(self.weights)
        self.weights += self.momentum * self.velocity
        step_direction = self.grads(views, u=self.weights)
        if self.line_search:
            self.learning_rate = self._backtracking_line_search(views, step_direction)
        step_size = self.learning_rate
        self.velocity = self.momentum * self.velocity - step_size * step_direction
        self.weights = self._gradient_step(self.weights, self.velocity)

    def _backtracking_line_search(self, views, step_direction, beta=0.9):
        # Initialize the gradient vector
        gradient = self.grads(views, u=self.weights)

        # Initialize the step size
        if self.momentum == 0:
            step_size = self.learning_rate
        else:
            step_size = 1

        # Perform backtracking line search until the objective function decreases
        while True:
            # Update the solution vector based on the gradient and step size
            updated_solution = self._gradient_step(
                self.weights, -step_direction * step_size
            )

            # Check if the objective function decreases with the updated solution
            if self.objective(views, u=updated_solution) < self.objective(
                views, u=self.weights
            ) - 0.1 * step_size * np.linalg.norm(step_direction.T @ gradient):
                break

            # Reduce the step size by a factor of beta
            step_size *= beta

            # Set a lower bound for the step size to avoid numerical issues
            if step_size < 1e-3:
                step_size = 1e-3
                break

        return step_size

    def grads(self, views, u=None):
        Aw, Bw, wAw, wBw = self._get_terms(views, u, unbiased=True)
        grads = 2 * Aw - (Aw @ np.triu(wBw) * np.sign(np.diag(wAw)) + Bw @ np.triu(wAw))
        return -grads

    def _Aw(self, views, projections):
        Aw = np.vstack(
            [
                np.cov(view.T, projections.sum(axis=0).T)[
                    0 : view.shape[1], view.shape[1] :
                ]
                for view in views
            ]
        )
        return Aw / len(views)

    def _Bw(self, views, projections, u):
        weights = self._split_weights(views, u)
        Bw = []
        for i, (view, projection, weight) in enumerate(
            zip(views, projections, weights)
        ):
            Bw.append((view.T @ projection) / (projection.shape[0] - 1))
        return np.vstack(Bw) / len(views)

    def _get_terms(self, views, u, projections=None, unbiased=False):
        if projections is None:
            projections = self.projections(views, u)
        if unbiased:
            #split views into two parts
            views1 = [view[:view.shape[0]//2] for view in views]
            views2 = [view[view.shape[0]//2:] for view in views]
        else:
            views1 = views
            views2 = views
        Aw = self._Aw(views1, projections)
        Bw = self._Bw(views2, projections, u)
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

    def _gradient_step(self, weights, velocity):
        return weights + velocity

    def _more_tags(self):
        return {"multiview": True, "stochastic": True}


class PLSEigenGame(CCAEigenGame):
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
        momentum=0,
        line_search=False,
        rho=0.1,
        ensure_descent=False,
        initialization="random",
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
            momentum=momentum,
            line_search=line_search,
            rho=rho,
            ensure_descent=ensure_descent,
            initialization=initialization,
        )

    def _Aw(self, views, projections):
        Aw = np.vstack(
            [
                np.cov(view, projections.sum(axis=0), rowvar=False)[
                    0 : view.shape[1], view.shape[1] :
                ]
                for view in views
            ]
        ) - np.vstack(
            [
                np.cov(view, projection, rowvar=False)[
                    0 : view.shape[1], view.shape[1] :
                ]
                for view, projection in zip(views, projections)
            ]
        )
        return Aw / len(views)

    def _Bw(self, views, projections, u):
        weights = self._split_weights(views, u)
        Bw = []
        for i, (view, projection, weight) in enumerate(
            zip(views, projections, weights)
        ):
            Bw.append(weight)
        return np.vstack(Bw) / len(views)
