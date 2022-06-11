import numpy as np

from ._base import _BaseStochastic


class IncrementalPLS(_BaseStochastic):
    r"""
    A class used to fit Incremental PLS

    :Maths:

    .. math::


    :Citation:

    Arora, Raman, et al. "Stochastic optimization for PCA and PLS." 2012 50th Annual Allerton Conference on Communication, Control, and Computing (Allerton). IEEE, 2012.

    :Example:

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
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        epochs=1,
        simple=False,
        val_interval=10,
    ):
        """
        Constructor for IncrementalPLS

        :param latent_dims: number of latent dimensions to fit
        :param scale: normalize variance in each column before fitting
        :param centre: demean data by column before fitting (and before transforming out of sample
        :param copy_data: If True, views will be copied; else, it may be overwritten
        :param random_state: Pass for reproducible output across multiple function calls
        :param accept_sparse: which forms are accepted for sparse data
        """
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
            val_interval=val_interval,
        )
        self.simple = simple

    def update(self, views):
        if not hasattr(self, "weights"):
            self.weights = [
                np.random.rand(view.shape[1], self.latent_dims) for view in views
            ]
        if not hasattr(self, "S"):
            self.S = np.zeros(self.latent_dims)
            self.count = 0
        if self.simple:
            self.simple_update(views)
        else:
            self.incremental_update(views)

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
        return np.sum(
            np.diag(
                np.cov(*self.transform(views), rowvar=False)[
                    : self.latent_dims, self.latent_dims :
                ]
            )
        )
