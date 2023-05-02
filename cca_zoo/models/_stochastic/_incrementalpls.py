from typing import Union

import numpy as np

from ._base import _BaseStochastic, tv


class IncrementalPLS(_BaseStochastic):
    r"""
    A class used to fit Incremental PLS

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
    simple : bool, optional
        Whether to use the simple update, by default False

    References
    ----------
    Arora, Raman, et al. "Stochastic optimization for PCA and PLS." 2012 50th Annual Allerton Conference on Communication, Control, and Computing (Allerton). IEEE, 2012.
    """

    def __init__(
        self,
        latent_dims: int = 1,
        scale: bool = True,
        centre=True,
        copy_data=True,
        random_state=None,
        accept_sparse=None,
        batch_size=1,
        shuffle=True,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        timeout=0,
        worker_init_fn=None,
        epochs=1,
        simple=False,
        initialization: Union[str, callable] = "random",
    ):
        super().__init__(
            latent_dims=latent_dims,
            scale=scale,
            centre=centre,
            copy_data=copy_data,
            accept_sparse=accept_sparse,
            random_state=random_state,
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
            initialization=initialization,
        )
        self.simple = simple

    def _update(self, views):
        if not hasattr(self, "S"):
            self.S = np.zeros(self.latent_dims)
            self.count = 0
        if self.simple:
            self.simple_update(views)
        else:
            self.incremental_update(views)
        return False

    def incremental_update(self, views):
        hats = np.stack([view @ weight for view, weight in zip(views, self.weights)])
        orths = [
            view - hat @ weight.T
            for view, weight, hat in zip(views, self.weights, hats)
        ]
        self.incrsvd(hats, orths)

    def simple_update(self, views):
        if not hasattr(self, "M"):
            self.M = np.zeros((views[0].shape[1], views[1].shape[1]))
        self.M = (
            views[0].T @ views[1]
            + self.weights[0] @ np.diag(self.S) @ self.weights[1].T
        )
        U, S, Vt = np.linalg.svd(self.M)
        self.weights[0] = U[:, : self.latent_dims]
        self.weights[1] = Vt.T[:, : self.latent_dims]
        self.S = S[: self.latent_dims]

    def incrsvd(self, hats, orths):
        Q = np.vstack(
            (
                np.hstack(
                    (
                        np.diag(self.S) + hats[0].T @ hats[1],
                        np.linalg.norm(orths[1], axis=1).T * hats[0].T,
                    )
                ),
                np.hstack(
                    (
                        (np.linalg.norm(orths[0], axis=1).T * hats[1].T).T,
                        np.atleast_2d(
                            np.linalg.norm(orths[0], axis=1, keepdims=True)
                            @ np.linalg.norm(orths[1], axis=1, keepdims=True).T
                        ),
                    )
                ),
            )
        )
        U, S, Vt = np.linalg.svd(Q)
        self.weights[0] = (
            np.hstack((self.weights[0], orths[0].T / np.linalg.norm(orths[0])))
            @ U[:, : self.latent_dims]
        )
        self.weights[1] = (
            np.hstack((self.weights[1], orths[1].T / np.linalg.norm(orths[1])))
            @ Vt.T[:, : self.latent_dims]
        )
        self.S = S[: self.latent_dims]

    def objective(self, views, **kwargs):
        q = [np.linalg.qr(weight)[0] for weight in self.weights]
        views = [self.scalers[i].transform(view) for i, view in enumerate(views)]
        transformed_views = []
        for i, (view) in enumerate(views):
            transformed_view = view @ q[i]
            transformed_views.append(transformed_view)
        return tv(transformed_views)
