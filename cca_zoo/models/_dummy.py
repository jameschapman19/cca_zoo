from typing import Iterable

import numpy as np

from cca_zoo.models._base import BaseModel


class DummyCCA(BaseModel):
    def __init__(
        self,
        latent_dims: int = 1,
        copy_data=True,
        random_state=None,
        accept_sparse=None,
        uniform=False,
    ):
        if accept_sparse is None:
            accept_sparse = ["csc", "csr"]
        super().__init__(
            latent_dims=latent_dims,
            copy_data=copy_data,
            accept_sparse=accept_sparse,
            random_state=random_state,
        )
        self.uniform = uniform

    def fit(self, views: Iterable[np.ndarray], y=None, **kwargs):
        self._validate_data(views)
        if self.uniform:
            self.weights = [
                np.ones((view.shape[1], self.latent_dims)) for view in views
            ]
        else:
            self.weights = [
                self.random_state.normal(0, 1, size=(view.shape[1], self.latent_dims))
                for view in views
            ]
        self.normalize_weights(views)
        return self

    def normalize_weights(self, views):
        self.weights = [
            weight
            / np.sqrt(np.diag(np.atleast_1d(np.cov(view @ weight, rowvar=False))))
            for view, weight in zip(views, self.weights)
        ]


class DummyPLS(DummyCCA):
    def normalize_weights(self, views):
        self.weights = [
            weight / np.linalg.norm(weight, axis=0)
            for view, weight in zip(views, self.weights)
        ]
