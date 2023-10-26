from typing import Iterable

import numpy as np
import tensorly as tl
from scipy.linalg import sqrtm
from tensorly.decomposition import parafac

from cca_zoo.linear._mcca import MCCA


class TCCA(MCCA):
    r"""
    A class used to fit TCCA model. This model extends MCCA to higher order correlations by using tensor products of the representations.

    The objective function of TCCA is:

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{ w_1^TX_1^T\otimes w_2^TX_2^T\otimes \cdots \otimes w_m^TX_m^Tw  \}\\

        \text{subject to:}

        w_i^TX_i^TX_iw_i=1

    where :math:`\otimes` denotes the Kronecker product.

    References
    ----------
    Kim, Tae-Kyun, Shu-Fai Wong, and Roberto Cipolla. "Tensor canonical correlation analysis for action classification." 2007 IEEE Conference on Computer Vision and Pattern Recognition. IEEE, 2007

    Examples
    --------
    >>> from cca_zoo.linear import TCCA
    >>> rng=np.random.RandomState(0)
    >>> X1 = rng.random((10,5))
    >>> X2 = rng.random((10,5))
    >>> X3 = rng.random((10,5))
    >>> model = TCCA()
    >>> model.fit((X1,X2,X3)).score((X1,X2,X3))
    """

    def fit(self, views: Iterable[np.ndarray], y=None, **kwargs):
        # Validate the input data
        views = self._validate_data(views)
        self._check_params()
        # returns whitened representations along with whitening matrices
        whitened_views, covs_invsqrt = self._setup_tensor(views)
        # The idea here is to form a matrix with M dimensions one for each view where at index
        # M[p_i,p_j,p_k...] we have the sum over n samples of the product of the pth feature of the
        # ith, jth, kth view etc.
        for i, el in enumerate(whitened_views):
            # To achieve this we start with the first view so M is nxp.
            if i == 0:
                M = el
            # For the remaining representations we expand their dimensions to match M i.e. nx1x...x1xp
            else:
                for _ in range(len(M.shape) - 1):
                    el = np.expand_dims(el, 1)
                # Then we perform an outer product by expanding the dimensionality of M and
                # outer product with the expanded el
                M = np.expand_dims(M, -1) @ el
        M = np.mean(M, 0)
        tl.set_backend("numpy")
        M_parafac = parafac(
            M,
            self.latent_dimensions,
            verbose=False,
            random_state=self.random_state,
        )
        self.weights_ = [
            cov_invsqrt @ fac
            for i, (cov_invsqrt, fac) in enumerate(zip(covs_invsqrt, M_parafac.factors))
        ]
        return self

    def correlations(self, views: Iterable[np.ndarray], **kwargs):
        """
        Predicts the correlation for the given data using the fit model

        :param views: list/tuple of numpy arrays or array likes with the same number of rows (samples)
        :param kwargs: any additional keyword arguments required by the given model
        """
        transformed_views = self.transform(views, **kwargs)
        transformed_views = [
            transformed_view - transformed_view.mean(axis=0)
            for transformed_view in transformed_views
        ]
        multiplied_views = np.stack(transformed_views, axis=0).prod(axis=0).sum(axis=0)
        norms = np.stack(
            [
                np.linalg.norm(transformed_view, axis=0)
                for transformed_view in transformed_views
            ],
            axis=0,
        ).prod(axis=0)
        corrs = multiplied_views / norms
        return corrs

    def average_pairwise_correlations(
        self, views: Iterable[np.ndarray], **kwargs
    ) -> np.ndarray:
        transformed_views = self.transform(views, **kwargs)
        transformed_views = [
            transformed_view - transformed_view.mean(axis=0)
            for transformed_view in transformed_views
        ]
        multiplied_views = np.stack(transformed_views, axis=0).prod(axis=0).sum(axis=0)
        norms = np.stack(
            [
                np.linalg.norm(transformed_view, axis=0)
                for transformed_view in transformed_views
            ],
            axis=0,
        ).prod(axis=0)
        corrs = multiplied_views / norms
        return corrs

    def score(self, views: Iterable[np.ndarray], **kwargs):
        return self.average_pairwise_correlations(views, **kwargs).mean()

    def _setup_tensor(self, views: Iterable[np.ndarray], **kwargs):
        covs = [
            (1 - self.c[i]) * np.cov(view, rowvar=False)
            + self.c[i] * np.eye(view.shape[1])
            for i, view in enumerate(views)
        ]
        smallest_eigs = [
            min(0, np.linalg.eigvalsh(cov).min()) - self.eps for cov in covs
        ]
        covs = [
            cov - smallest_eig * np.eye(cov.shape[0])
            for cov, smallest_eig in zip(covs, smallest_eigs)
        ]
        covs_invsqrt = [np.linalg.inv(sqrtm(cov).real) for cov in covs]
        views = [
            train_view @ cov_invsqrt
            for train_view, cov_invsqrt in zip(views, covs_invsqrt)
        ]
        return views, covs_invsqrt

    def _more_tags(self):
        return {"multiview": True}
