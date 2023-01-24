import warnings
from typing import Iterable

import numpy as np
from scipy.linalg import block_diag

from cca_zoo.models._multiview._mcca import MCCA


class PRCCA(MCCA):
    """
    Partially Regularized Canonical Correlation Analysis


    Parameters
    ----------
    latent_dims : int, optional
        Number of latent dimensions to use, by default 1
    scale : bool, optional
        Whether to scale the data, by default True
    centre : bool, optional
        Whether to centre the data, by default True
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
        latent_dims: int = 1,
        scale: bool = True,
        centre=True,
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
            latent_dims=latent_dims,
            scale=scale,
            centre=centre,
            copy_data=copy_data,
            random_state=random_state,
            eps=eps,
            c=c,
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
        if idxs is None:
            warnings.warn(f"No idxs provided, using all features")
            idxs = [np.arange(views[0].shape[1], dtype=int)] * len(views)
        for idx in idxs:
            assert idx.dtype == int, "subject_groups must be integers"
        views = self._validate_inputs(views)
        self._check_params()
        views = self._pretransform(views, idxs)
        super().fit(views, idxs=idxs)
        self._transform_weights(views, idxs=idxs)
        return self

    def _pretransform(self, views, idxs):
        X_1 = [view[:, idx] for view, idx in zip(views, idxs)]
        self.p = [X_i.shape[1] for X_i in X_1]
        X_2 = [np.delete(view, idx, axis=1) for view, idx in zip(views, idxs)]
        self.B = [np.linalg.pinv(X_2) @ X_1 for X_1, X_2 in zip(X_1, X_2)]
        X_1 = [X_1 - X_2 @ B for X_1, X_2, B in zip(X_1, X_2, self.B)]
        views = [np.hstack((X_1, X_2)) for X_1, X_2 in zip(X_1, X_2)]
        return views

    def _setup_evp(self, views: Iterable[np.ndarray], idxs=None, **kwargs):
        all_views = np.concatenate(views, axis=1)
        C = all_views.T @ all_views / self.n
        penalties = [np.zeros((view.shape[1])) for view in views]
        for i, idx in enumerate(idxs):
            penalties[i][idx] = self.c[i]
        D = block_diag(
            *[
                (1 - self.c[i]) * (m.T @ m) / self.n + np.diag(penalties[i])
                for i, m in enumerate(views)
            ]
        )
        C -= block_diag(*[view.T @ view / self.n for view in views])
        D_smallest_eig = min(0, np.linalg.eigvalsh(D).min()) - self.eps
        D = D - D_smallest_eig * np.eye(D.shape[0])
        self.splits = np.cumsum([0] + [view.shape[1] for view in views])
        return C, D

    def _transform_weights(self, views, idxs=None):
        for i, idx in enumerate(idxs):
            alpha_1 = self.weights[i][idx]
            alpha_2 = np.delete(self.weights[i], idx, axis=0)
            alpha_2 -= self.B[i] @ alpha_1
            mask = np.ones(self.weights[i].shape[0], dtype=bool)
            mask[idx] = False
            self.weights[i][mask] = alpha_2
