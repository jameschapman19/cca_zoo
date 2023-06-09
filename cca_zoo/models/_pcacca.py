from typing import Iterable, Union

from sklearn.decomposition import PCA
import numpy as np
from sklearn.utils.validation import check_is_fitted

from cca_zoo.models._mcca import MCCA


class PCACCA(MCCA):
    """
    Principal Component Analysis CCA

    Data driven PCA on each view followed by CCA on the PCA components. Keep percentage of variance
    """

    def __init__(
        self,
        latent_dims=1,
        copy_data=True,
        random_state=None,
        c: Union[Iterable[float], float] = None,
        eps=1e-9,
        percent_variance=0.99,
    ):
        super().__init__(
            latent_dims=latent_dims,
            copy_data=copy_data,
            random_state=random_state,
            c=c,
            eps=eps,
        )
        self.percent_variance = percent_variance

    def fit(self, views: Iterable[np.ndarray], y=None, **kwargs):
        # Validate the input data
        self._validate_data(views)
        # Check the parameters
        self._check_params()
        # Do data driven PCA on each view
        pca_views = self._pca(views)
        self._validate_data(pca_views)
        # Setup the eigenvalue problem
        C, D = self._setup_evp(pca_views, **kwargs)
        # Solve the eigenvalue problem
        eigvals, eigvecs = self._solve_evp(C, D)
        # Compute the weights for each view
        self._weights(eigvals, eigvecs, pca_views)
        # Transform the weights back to the original space
        self.transform_weights()
        return self

    def _pca(self, views):
        """
        Do data driven PCA on each view
        """
        self.pca = [PCA() for _ in views]
        # Fit PCA on each view and then keep the components that explain the percentage of variance
        for i, view in enumerate(views):
            self.pca[i].fit(view)
            # Keep the components that explain the percentage of variance
            explained_variance = self.pca[i].explained_variance_ratio_
            n_components_ = (
                np.where(np.cumsum(explained_variance) >= self.percent_variance)[0][0]
                + 1
            )
            self.pca[i].n_components_ = n_components_
            self.pca[i].components_ = self.pca[i].components_[:n_components_]
        return [self.pca[i].transform(view) for i, view in enumerate(views)]

    def transform_weights(self):
        # go from weights in PCA space to weights in original space
        self.weights = [
            pca.components_.T @ self.weights[i]
            for i, pca in enumerate(self.pca)
        ]

if __name__ == '__main__':
    from cca_zoo.models import rCCA
    x = np.random.rand(100, 10)
    y = np.random.rand(100, 10)
    x -= x.mean(axis=0)
    y -= y.mean(axis=0)
    model = rCCA(latent_dims=2, c=0.1)
    model.fit([x, y])
    print(model.weights)
    print(model.score([x, y]))
    # compare to normal CCA
    from cca_zoo.models import CCA

    model = PCACCA(latent_dims=2, c=0.1)
    model.fit([x, y])
    print(model.weights)
    print(model.score([x, y]))
    print()
