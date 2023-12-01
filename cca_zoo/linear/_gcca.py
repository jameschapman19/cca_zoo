from typing import Iterable, Union

import numpy as np

from cca_zoo._utils._checks import _process_parameter
from cca_zoo.linear._mcca import MCCA


class GCCA(MCCA):
    r"""
    A class used to fit GCCA model. This model extends CCA to more than two representations by optimizing the sum of correlations with a shared auxiliary vector.

    The objective function of GCCA is:

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{ \sum_iw_i^TX_i^TT  \}\\

        \text{subject to:}

        T^TT=1

    where :math:`T` is the auxiliary vector.

    Examples
    --------
    >>> from cca_zoo.linear import GCCA
    >>> import numpy as np
    >>> rng=np.random.RandomState(0)
    >>> X1 = rng.random((10,5))
    >>> X2 = rng.random((10,5))
    >>> X3 = rng.random((10,5))
    >>> model = GCCA()
    >>> model.fit((X1,X2,X3)).score((X1,X2,X3))

    References
    ----------
    Tenenhaus, Arthur, and Michel Tenenhaus. "Regularized generalized canonical correlation analysis." Psychometrika 76.2 (2011): 257.
    """

    def __init__(
        self,
        latent_dimensions: int = 1,
        copy_data=True,
        random_state=None,
        c: Union[Iterable[float], float] = None,
        view_weights: Iterable[float] = None,
        eps: float = 1e-6,
    ):
        super().__init__(
            latent_dimensions=latent_dimensions,
            copy_data=copy_data,
            accept_sparse=["csc", "csr"],
            random_state=random_state,
            c=c,
            eps=eps,
            pca=False,
        )
        self.view_weights = view_weights

    def fit(self, views: Iterable[np.ndarray], y=None, K=None, **kwargs):
        return super().fit(views, y=y, K=K, **kwargs)

    def _check_params(self):
        self.c = _process_parameter("c", self.c, 0, self.n_views_)

    def _C(self, views, K=None):
        if K is None:
            # just use identity when all rows are observed in all representations.
            K = np.ones((len(views), views[0].shape[0]))
        Q = []
        self.view_weights = _process_parameter(
            "view_weights", self.view_weights, 1, self.n_views_
        )
        for i, (view, view_weight) in enumerate(zip(views, self.view_weights)):
            view_cov = (1 - self.c[i]) * np.cov(view, rowvar=False) + self.c[
                i
            ] * np.eye(view.shape[1])
            smallest_eig = min(0, np.linalg.eigvalsh(view_cov).min()) - self.eps
            view_cov = view_cov - smallest_eig * np.eye(view_cov.shape[0])
            Q.append(view_weight * view @ np.linalg.inv(view_cov) @ view.T)
        Q = np.sum(Q, axis=0)
        Q = (
            np.diag(np.sqrt(np.sum(K, axis=0)))
            @ Q
            @ np.diag(np.sqrt(np.sum(K, axis=0)))
        )
        return Q

    def _D(self, views, **kwargs):
        return None

    def _weights(self, eigvals, eigvecs, views, **kwargs):
        self.weights_ = [
            np.linalg.pinv(view) @ eigvecs[:, : self.latent_dimensions]
            for view in views
        ]

    def _more_tags(self):
        return {"multiview": True}
