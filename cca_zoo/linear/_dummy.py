from typing import Iterable

import numpy as np
from sklearn.utils import check_random_state

from cca_zoo._base import _BaseModel


class DummyCCA(_BaseModel):
    def __init__(
        self,
        latent_dimensions: int = 1,
        copy_data=True,
        random_state=None,
        accept_sparse=None,
        uniform=False,
    ):
        if accept_sparse is None:
            accept_sparse = ["csc", "csr"]
        super().__init__(
            latent_dimensions=latent_dimensions,
            copy_data=copy_data,
            accept_sparse=accept_sparse,
            random_state=random_state,
        )
        self.uniform = uniform

    def fit(self, views: Iterable[np.ndarray], y=None, **kwargs):
        self._validate_data(views)
        self.random_state = check_random_state(self.random_state)
        if self.uniform:
            self.weights_ = [
                np.ones((view.shape[1], self.latent_dimensions)) for view in views
            ]
        else:
            self.weights_ = [
                self.random_state.normal(
                    0, 1, size=(view.shape[1], self.latent_dimensions)
                )
                for view in views
            ]
        self.normalize_weights(views)
        return self

    def normalize_weights(self, views):
        self.weights_ = [
            weight / np.linalg.norm(weight, axis=0)
            for view, weight in zip(views, self.weights_)
        ]


class DummyPLS(DummyCCA):
    def normalize_weights(self, views):
        self.weights = [
            weight / np.linalg.norm(weight, axis=0)
            for view, weight in zip(views, self.weights_)
        ]
