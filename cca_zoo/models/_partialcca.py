from typing import Iterable, Union

import numpy as np
from sklearn.utils.validation import check_is_fitted

from cca_zoo.models import MCCA


class PartialCCA(MCCA):
    r"""
    A class used to fit a partial cca model. The key difference between this and a vanilla CCA or MCCA is that
    the canonical score vectors must be orthogonal to the supplied confounding variables.

    Parameters
    ----------
    latent_dims : int, optional
        The number of latent dimensions to use, by default 1
    copy_data : bool, optional
        Whether to copy the data, by default True
    random_state : int, optional
        The random state to use, by default None
    c : Union[Iterable[float], float], optional
        The regularization parameter, by default None

    References
    ----------
    Rao, B. Raja. "Partial canonical correlations." Trabajos de estadistica y de investigación operativa 20.2-3 (1969): 211-219.

    Example
    -------
    >>> from cca_zoo.models import PartialCCA
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> partials = np.random.rand(10,3)
    >>> model = PartialCCA()
    >>> model.fit((X1,X2),partials=partials).score((X1,X2))
    array([0.99993046])

    """


    def __init__(
        self,
        latent_dims: int = 1,
        copy_data=True,
        random_state=None,
        c: Union[Iterable[float], float] = None,
    ):
        super().__init__(
            latent_dims=latent_dims,
            copy_data=copy_data,
            random_state=random_state,
        )
        self.c = c # store the regularization parameter

    def fit(self, views: Iterable[np.ndarray], y=None, partials=None, **kwargs):
        super().fit(views, y=y, partials=partials, **kwargs) # call the parent class fit method

    def _setup_evp(self, views: Iterable[np.ndarray], partials=None, **kwargs):
        if partials is None:
            raise ValueError(
                f"partials is {partials}. Require matching partials to transform with"
                f"partial CCA."
            )
        self.confound_betas = [np.linalg.pinv(partials) @ view for view in views] # compute the confounding betas for each view using pseudo-inverse of partials
        views = [
            view - partials @ np.linalg.pinv(partials) @ view # remove the confounding effect from each view using projection matrix
            for view, confound_beta in zip(views, self.confound_betas)
        ]
        return super()._setup_evp(views) # call the parent class _setup_evp method

    def transform(self, views: Iterable[np.ndarray], partials=None, **kwargs):
        if partials is None:
            raise ValueError(
                f"partials is {partials}. Require matching partials to transform with"
                f"partial CCA."
            )
        check_is_fitted(self, attributes=["weights"]) # check if the model has been fitted before transforming
        transformed_views = []
        for i, (view) in enumerate(views):
            transformed_view = (
                view - partials @ self.confound_betas[i] # remove the confounding effect from each view using stored confounding betas
            ) @ self.weights[i] # multiply each view by its corresponding weight matrix
            transformed_views.append(transformed_view) # append the transformed view to the list of transformed views
        return transformed_views # return the list of transformed views

    def _more_tags(self):
        return {"multiview": True} # indicate that this model can handle multiview data