from typing import Iterable

import numpy as np

from cca_zoo.models._base import _BaseCCA


class _DummyCCA(_BaseCCA):
    def __init__(
        self,
        latent_dims: int = 1,
        scale: bool = True,
        centre=True,
        copy_data=True,
        random_state=None,
        accept_sparse=None,
        uniform=False,
    ):
        if accept_sparse is None:
            accept_sparse = ["csc", "csr"]
        super().__init__(
            latent_dims=latent_dims,
            scale=scale,
            centre=centre,
            copy_data=copy_data,
            accept_sparse=accept_sparse,
            random_state=random_state,
        )
        self.uniform = uniform

    def fit(self, views: Iterable[np.ndarray], y=None, **kwargs):
        views = self._validate_inputs(views)
        if self.uniform:
            weights = [np.ones((view.shape[1], self.latent_dims)) for view in views]
        else:
            weights = [
                self.random_state.normal(0, 1, size=(view.shape[1], self.latent_dims))
                for view in views
            ]
        self.weights = [weight / np.linalg.norm(weight, axis=0) for weight in weights]
        return self
