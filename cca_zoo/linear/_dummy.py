from typing import Iterable

import numpy as np

from cca_zoo._base import BaseModel


class DummyCCA(BaseModel):
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
        if self.uniform:
            self.weights = [
                np.ones((view.shape[1], self.latent_dimensions)) for view in views
            ]
        else:
            self.weights = [
                self.random_state.normal(
                    0, 1, size=(view.shape[1], self.latent_dimensions)
                )
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
        scores = self.average_pairwise_correlations(views)
        for i, score in enumerate(scores):
            if score < 0:
                # flip the sign of the first weights
                self.weights[0][:, i] = -self.weights[0][:, i]


class DummyPLS(DummyCCA):
    def normalize_weights(self, views):
        self.weights = [
            weight / np.linalg.norm(weight, axis=0)
            for view, weight in zip(views, self.weights)
        ]
