import warnings
from typing import Iterable

import numpy as np
from scipy.linalg import block_diag

from cca_zoo.linear._mcca import MCCA


class PRCCA(MCCA):
    """
    Partially Regularized Canonical Correlation Analysis


    Parameters
    ----------
    latent_dimensions : int, optional
        Number of latent dimensions to use, by default 1
    copy_data : bool, optional
        Whether to copy the data, by default True
    random_state : int, optional
        Random state for reproducibility, by default None
    eps : float, optional
        Tolerance for convergence, by default 1e-3
    c : Union[Iterable[float], float], optional
        Regularisation parameter, by default None

    References
    ----------
    Tuzhilina, Elena, Leonardo Tozzi, and Trevor Hastie. "Canonical correlation analysis in high dimensions with structured regularization." Statistical Modelling (2021): 1471082X211041033.
    """

    def __init__(
        self,
        latent_dimensions: int = 1,
        copy_data=True,
        random_state=None,
        eps=1e-3,
        c=0,
    ):
        """
        Parameters
        ----------
        c : Union[Iterable[float], float], optional
            Regularisation parameter, by default None
        eps : float, optional
            Tolerance for convergence, by default 1e-3
        """
        super().__init__(
            latent_dimensions=latent_dimensions,
            copy_data=copy_data,
            random_state=random_state,
            eps=eps,
            c=c,
            pca=False,
        )

    def fit(self, views: Iterable[np.ndarray], y=None, idxs=None, **kwargs):
        """
        Parameters
        ----------
        views : list/tuple of numpy arrays or array likes with the same number of rows (samples)
        y : None
        idxs : list/tuple of integers indicating which features from each view are the partially regularised features
        kwargs: any additional keyword arguments required by the given model

        """
        # Validate the input data
        views = self._validate_data(views)
        # Check the parameters
        self._check_params()
        if idxs is None:
            warnings.warn("No idxs provided, using all features")
            idxs = [np.arange(views[0].shape[1], dtype=int)] * self.n_views_
        for idx in idxs:
            assert np.issubdtype(
                idx.dtype, np.integer
            ), "feature groups must be integers"
        return super().fit(views, y=y, idxs=idxs, **kwargs)

    def _process_data(self, views, idxs=None, **kwargs):
        X_1 = [view[:, idx] for view, idx in zip(views, idxs)]
        self.p = [X_i.shape[1] for X_i in X_1]
        X_2 = [np.delete(view, idx, axis=1) for view, idx in zip(views, idxs)]
        self.B = [np.linalg.pinv(X_2) @ X_1 for X_1, X_2 in zip(X_1, X_2)]
        X_1 = [X_1 - X_2 @ B for X_1, X_2, B in zip(X_1, X_2, self.B)]
        views = [np.hstack((X_1, X_2)) for X_1, X_2 in zip(X_1, X_2)]
        return views

    def _C(self, views, **kwargs):
        all_views = np.concatenate(views, axis=1)
        C = np.cov(all_views, rowvar=False)
        C -= block_diag(*[np.cov(view, rowvar=False) for view in views])
        return C

    def _D(self, views: Iterable[np.ndarray], idxs=None, **kwargs):
        penalties = [np.zeros((view.shape[1])) for view in views]
        for i, idx in enumerate(idxs):
            penalties[i][idx] = self.c[i]
        D = block_diag(
            *[
                (1 - self.c[i]) * np.cov(view, rowvar=False) + np.diag(penalties[i])
                for i, view in enumerate(views)
            ]
        )
        D_smallest_eig = min(0, np.linalg.eigvalsh(D).min()) - self.eps
        D = D - D_smallest_eig * np.eye(D.shape[0])
        return D

    def _weights(self, eigvals, eigvecs, views, idxs=None, **kwargs):
        # split eigvecs into weights_ for each view
        self.weights_ = np.split(eigvecs, self.splits[:-1], axis=0)
        for i, idx in enumerate(idxs):
            alpha_1 = self.weights_[i][idx]
            alpha_2 = np.delete(self.weights_[i], idx, axis=0)
            alpha_2 -= self.B[i] @ alpha_1
            mask = np.ones(self.weights_[i].shape[0], dtype=bool)
            mask[idx] = False
            self.weights_[i][mask] = alpha_2

    def _more_tags(self):
        return {"multiview": True}
