from typing import Iterable, Union

import numpy as np

from cca_zoo.linear._mcca import MCCA


class PCACCA(MCCA):
    """
    Principal Component Analysis CCA

    Data driven PCA on each view followed by CCA on the PCA components. Keep percentage of variance

    Examples
    --------
    >>> from cca_zoo.linear import PCACCA
    >>> import numpy as np
    >>> X1 = np.random.rand(100, 5)
    >>> X2 = np.random.rand(100, 5)
    >>> model = PCACCA(latent_dimensions=1, percent_variance=0.5)
    >>> model.fit([X1, X2])
    """

    def __init__(
        self,
        latent_dimensions=1,
        copy_data=True,
        random_state=None,
        c: Union[Iterable[float], float] = None,
        eps=1e-9,
        percent_variance=0.99,
    ):
        super().__init__(
            latent_dimensions=latent_dimensions,
            copy_data=copy_data,
            random_state=random_state,
            c=c,
            eps=eps,
        )
        # ensure that the percent variance is less than 1 and give a warning if not
        assert percent_variance <= 1, "percent_variance must be less than 1"
        self.percent_variance = percent_variance

    def _process_data(self, views, **kwargs):
        views = self._apply_pca(views)
        for i, view in enumerate(views):
            # Keep the components that explain the percentage of variance
            explained_variance = self.pca_models[i].explained_variance_ratio_
            n_components_ = (
                np.where(np.cumsum(explained_variance) >= self.percent_variance)[0][0]
                + 1
            )
            self.pca_models[i].n_components_ = n_components_
            self.pca_models[i].components_ = self.pca_models[i].components_[
                :n_components_
            ]
            self.pca_models[i].explained_variance_ = self.pca_models[
                i
            ].explained_variance_[:n_components_]
            self.pca_models[i].explained_variance_ratio_ = self.pca_models[
                i
            ].explained_variance_ratio_[:n_components_]
            self.pca_models[i].singular_values_ = self.pca_models[i].singular_values_[
                :n_components_
            ]
        return [
            view[:, : self.pca_models[i].n_components_] for i, view in enumerate(views)
        ]
