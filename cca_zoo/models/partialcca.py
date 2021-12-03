from typing import Iterable, Union

import numpy as np
from scipy.linalg import block_diag
from sklearn.utils.validation import check_is_fitted

from cca_zoo.models import MCCA
from cca_zoo.utils import _check_views


class PartialCCA(MCCA):
    r"""
    A class used to fit a partial cca model. The key difference between this and a vanilla CCA or MCCA is that
    the canonical score vectors must be orthogonal to the supplied confounding variables.

    :Citation:

    Rao, B. Raja. "Partial canonical correlations." Trabajos de estadistica y de investigación operativa 20.2-3 (1969): 211-219.

    :Example:
    >>> from cca_zoo.models import PartialCCA
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> confounds = np.random.rand(10,3)
    >>> model = PartialCCA()
    >>> model.fit((X1,X2),confounds=confounds).score((X1,X2),confounds=confounds)
    array([0.99993046])

    """

    def __init__(
        self,
        latent_dims: int = 1,
        scale: bool = True,
        centre=True,
        copy_data=True,
        random_state=None,
        c: Union[Iterable[float], float] = None,
        eps=1e-3,
    ):
        """
        Constructor for Partial CCA

        :param latent_dims: number of latent dimensions to fit
        :param scale: normalize variance in each column before fitting
        :param centre: demean data by column before fitting (and before transforming out of sample
        :param copy_data: If True, X will be copied; else, it may be overwritten
        :param random_state: Pass for reproducible output across multiple function calls
        :param c: Iterable of regularisation parameters for each view (between 0:CCA and 1:PLS)
        :param eps: epsilon for stability
        """
        super().__init__(
            latent_dims=latent_dims,
            scale=scale,
            centre=centre,
            copy_data=copy_data,
            random_state=random_state,
        )
        self.c = c
        self.eps = eps

    def _setup_evp(self, views: Iterable[np.ndarray], confounds=None):
        if confounds is None:
            raise ValueError(
                f"confounds is {confounds}. Require matching confounds to transform with"
                f"partial CCA."
            )
        self.confound_betas = [np.linalg.pinv(confounds) @ view for view in views]
        views = [
            view - confounds @ np.linalg.pinv(confounds) @ view
            for view, confound_beta in zip(views, self.confound_betas)
        ]
        all_views = np.concatenate(views, axis=1)
        C = all_views.T @ all_views / self.n
        # Can regularise by adding to diagonal
        D = block_diag(
            *[
                (1 - self.c[i]) * m.T @ m / self.n + self.c[i] * np.eye(m.shape[1])
                for i, m in enumerate(views)
            ]
        )
        C -= block_diag(*[view.T @ view / self.n for view in views]) - D
        D_smallest_eig = min(0, np.linalg.eigvalsh(D).min()) - self.eps
        D = D - D_smallest_eig * np.eye(D.shape[0])
        self.splits = np.cumsum([0] + [view.shape[1] for view in views])
        return views, C, D

    # TODO TRANSFORM
    def transform(self, views: Iterable[np.ndarray], y=None, confounds=None):
        """
        Transforms data given a fit model

        :param views: numpy arrays with the same number of rows (samples) separated by commas
        """
        if confounds is None:
            raise ValueError(
                f"confounds is {confounds}. Require matching confounds to transform with"
                f"partial CCA."
            )
        check_is_fitted(self, attributes=["weights"])
        views = _check_views(
            *views, copy=self.copy_data, accept_sparse=self.accept_sparse
        )
        views = self._centre_scale_transform(views)
        transformed_views = []
        for i, (view) in enumerate(views):
            transformed_view = (
                view - confounds @ self.confound_betas[i]
            ) @ self.weights[i]
            transformed_views.append(transformed_view)
        return transformed_views


def main():
    import numpy as np

    n = 100
    p = 10
    q = 10
    Z = np.random.rand(100, 1)
    Z -= Z.mean(axis=0)
    C = np.random.rand(100, 1)
    C -= C.mean(axis=0)
    Wx = np.random.rand(1, p)
    Wxc = np.random.rand(1, p)
    Wy = np.random.rand(1, q)
    Wyc = np.random.rand(1, q)
    X = Z @ Wx + C @ Wxc + 2 * np.random.rand(n, p)
    Y = Z @ Wy + C @ Wyc + 2 * np.random.rand(n, q)
    pcca = PartialCCA(centre=False, scale=False).fit((X, Y), confounds=C)
    a = pcca.score((X, Y), confounds=C)
    from cca_zoo.models import MCCA

    cca = MCCA(centre=False, scale=False).fit((X, Y), confounds=C)
    b = cca.score((X, Y))
    bb = np.corrcoef(cca.transform((X, Y))[0], C, rowvar=False)
    X_deconf = (np.eye(n) - C @ np.linalg.inv(C.T @ C) @ C.T) @ X
    Y_deconf = (np.eye(n) - C @ np.linalg.inv(C.T @ C) @ C.T) @ Y
    m = MCCA(centre=False, scale=False).fit((X_deconf, Y_deconf))
    c = m.score((X_deconf, Y_deconf))
    cc = np.corrcoef(m.transform((X_deconf, Y_deconf))[0], C, rowvar=False)
    aa = np.corrcoef(pcca.transform((X, Y), confounds=C)[0], C, rowvar=False)
    print()


if __name__ == "__main__":
    main()
