import numpy as np

from cca_zoo.models._stochastic._base import _BaseStochastic


class PLSStochasticPower(_BaseStochastic):
    r"""
    A class used to fit Stochastic PLS

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
            learning_rate=0.01,
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
        )

    def update(self, views):
        projections = np.stack(
            [view @ weight for view, weight in zip(views, self.weights)]
        )
        for i, view in enumerate(views):
            projections = np.ma.array(projections, mask=False, keep_mask=False)
            projections.mask[i] = True
            self.weights[i] += (
                    self.learning_rate * (view.T @ projections.sum(axis=0).filled()) / view.shape[0]
            )
        self.weights = [
            weight / np.linalg.norm(weight, axis=0) for weight in self.weights
        ]

    def objective(self, views, **kwargs):
        return self.tv(views)
