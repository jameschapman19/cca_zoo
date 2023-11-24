from typing import Iterable

import numpy as np
from sklearn.utils.validation import check_is_fitted

from cca_zoo.linear._mcca import MCCA


class PartialCCA(MCCA):
    r"""
    A class used to fit a partial CCA model. This model extends CCA to account for confounding variables that may affect the correlation between representations.

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{ w_1^TX_1^TX_2w_2  \}\\

        \text{subject to:}

        w_i^TX_i^TX_iw_i=1

        w_i^TX_i^TZ=0


    Example
    -------
    >>> from cca_zoo.linear import PartialCCA
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> partials = np.random.rand(10,3)
    >>> model = PartialCCA()
    >>> model.fit((X1,X2),partials=partials).score((X1,X2))

    References
    ----------
    Rao, B. Raja. "Partial canonical correlations." Trabajos de estadistica y de investigación operativa 20.2-3 (1969): 211-219.

    """

    def fit(self, views: Iterable[np.ndarray], y=None, partials=None, **kwargs):
        self.pca = False
        return super().fit(
            views, y=y, partials=partials, **kwargs
        )  # call the parent class fit method

    def _process_data(self, views, partials=None, **kwargs):
        if partials is None:
            return super()._process_data(views, **kwargs)
        else:
            self.confound_betas = [
                np.linalg.pinv(partials) @ view for view in views
            ]  # compute the confounding betas for each view using pseudo-inverse of partials
            views = [
                view
                - partials
                @ np.linalg.pinv(partials)
                @ view  # remove the confounding effect from each view using projection matrix
                for view, confound_beta in zip(views, self.confound_betas)
            ]
            return views

    def transform(self, views: Iterable[np.ndarray], partials=None, **kwargs):
        if partials is None:
            return super().transform(views, **kwargs)
        else:
            check_is_fitted(
                self
            )  # check if the model has been fitted before transforming
            transformed_views = []
            for i, (view) in enumerate(views):
                transformed_view = (
                    view
                    - partials @ self.confound_betas[i]
                    # remove the confounding effect from each view using stored confounding betas
                ) @ self.weights_[
                    i
                ]  # multiply each view by its corresponding weight matrix
                transformed_views.append(
                    transformed_view
                )  # append the transformed view to the list of transformed representations
            return transformed_views  # return the list of transformed representations

    def _more_tags(self):
        return {"multiview": True}  # indicate that this model can handle multiview data
