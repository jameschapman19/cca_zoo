import numpy as np

from ._base import _BaseStochastic


class StochasticPowerPLS(_BaseStochastic):
    r"""
    A class used to fit Stochastic PLS

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
        lr=1e-2,
    ):
        """
        Constructor for StochasticPowerPLS

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
        )
        self.lr = lr

    def update(self, views):
        if not hasattr(self, "weights"):
            self.weights = [
                np.random.rand(view.shape[1], self.latent_dims) for view in views
            ]
        projections = np.stack(
            [view @ weight for view, weight in zip(views, self.weights)]
        )
        for i, view in enumerate(views):
            projections = np.ma.array(projections, mask=False, keep_mask=False)
            projections.mask[i] = True
            self.weights[i] += (
                self.lr * (view.T @ projections.sum(axis=0).filled()) / view.shape[0]
            )
        self.weights = [
            weight / np.linalg.norm(weight, axis=0) for weight in self.weights
        ]

    def objective(self, views, **kwargs):
        return np.sum(
            np.diag(
                np.cov(*self.transform(views), rowvar=False)[
                    : self.latent_dims, self.latent_dims :
                ]
            )
        )
