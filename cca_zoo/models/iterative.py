import copy
import warnings
from abc import abstractmethod
from typing import Union, Iterable

import numpy as np

from .cca_base import _CCA_Base
from .innerloop import PLSInnerLoop, PMDInnerLoop, ParkhomenkoInnerLoop, ElasticInnerLoop, ADMMInnerLoop, \
    SpanCCAInnerLoop, SWCCAInnerLoop
from ..utils import check_views


class _Iterative(_CCA_Base):
    """
    A class used as the base for iterative CCA methods i.e. those that optimize for each dimension one at a time with deflation.

    """

    def __init__(self, latent_dims: int = 1, scale: bool = True, centre=True, copy_data=True, random_state=None,
                 deflation='cca',
                 max_iter: int = 100,
                 generalized: bool = False,
                 initialization: str = 'unregularized', tol: float = 1e-9):
        """
        Constructor for _Iterative

        :param latent_dims: number of latent dimensions to fit
        :param scale: normalize variance in each column before fitting
        :param centre: demean data by column before fitting (and before transforming out of sample
        :param copy_data: If True, X will be copied; else, it may be overwritten
        :param random_state: Pass for reproducible output across multiple function calls
        :param deflation: the type of deflation.
        :param max_iter: the maximum number of iterations to perform in the inner optimization loop
        :param generalized: use auxiliary variables (required for >2 views)
        :param initialization: intialization for optimisation. 'unregularized' uses CCA or PLS solution,'random' uses random initialization,'uniform' uses uniform initialization of weights and scores
        :param tol: tolerance value used for early stopping
        """
        super().__init__(latent_dims=latent_dims, scale=scale, centre=centre, copy_data=copy_data,
                         accept_sparse=['csc', 'csr'])
        self.max_iter = max_iter
        self.generalized = generalized
        self.initialization = initialization
        self.tol = tol
        self.deflation = deflation
        self.random_state = random_state

    def fit(self, *views: np.ndarray, ):
        """
        Fits the model by running an inner loop to convergence and then using deflation (currently only supports CCA deflation)

        :param views: numpy arrays with the same number of rows (samples) separated by commas
        """
        views = check_views(*views, copy=self.copy_data, accept_sparse=self.accept_sparse)
        views = self._centre_scale(*views)
        self.n_views = len(views)
        self.n = views[0].shape[0]
        self._set_loop_params()
        n = views[0].shape[0]
        p = [view.shape[1] for view in views]
        # List of d: p x k
        self.weights = [np.zeros((p_, self.latent_dims)) for p_ in p]
        self.loadings = [np.zeros((p_, self.latent_dims)) for p_ in p]

        # List of d: n x k
        self.scores = [np.zeros((n, self.latent_dims)) for _ in views]

        residuals = copy.deepcopy(list(views))

        self.objective = []
        # For each of the dimensions
        for k in range(self.latent_dims):
            self.loop = self.loop.fit(*residuals)
            for i, residual in enumerate(residuals):
                self.weights[i][:, k] = self.loop.weights[i].ravel()
                self.scores[i][:, k] = self.loop.scores[i].ravel()
                self.loadings[i][:, k] = np.dot(self.scores[i][:, k], residual)
                residuals[i] = self._deflate(residuals[i], self.scores[i][:, k], self.weights[i][:, k])
            self.objective.append(self.loop.track_objective)
        return self

    def _deflate(self, residual, score, loading):
        """
        Deflate view residual by CCA deflation (https://ars.els-cdn.com/content/image/1-s2.0-S0006322319319183-mmc1.pdf)

        :param residual:
        :param score:

        """
        if self.deflation == 'cca':
            return residual - np.outer(score, score) @ residual / np.dot(score, score).item()
        elif self.deflation == 'pls':
            return residual - np.outer(score, loading)

    @abstractmethod
    def _set_loop_params(self):
        """
        Sets up the inner optimization loop for the method. By default uses the PLS inner loop.
        """
        self.loop = PLSInnerLoop(max_iter=self.max_iter, generalized=self.generalized,
                                 initialization=self.initialization, random_state=self.random_state)


class PLS_ALS(_Iterative):
    """
    A class used to fit a PLS model

    Fits a partial least squares model with CCA deflation by NIPALS algorithm

    :Example:

    >>> from cca_zoo.models import PLS
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> model = PLS()
    >>> model.fit(X1,X2)
    """

    def __init__(self, latent_dims: int = 1, scale: bool = True, centre=True, copy_data=True, random_state=None,
                 max_iter: int = 100,
                 generalized: bool = False,
                 initialization: str = 'unregularized', tol: float = 1e-9):
        """
        Constructor for PLS

        :param latent_dims: number of latent dimensions to fit
        :param scale: normalize variance in each column before fitting
        :param centre: demean data by column before fitting (and before transforming out of sample
        :param copy_data: If True, X will be copied; else, it may be overwritten
        :param random_state: Pass for reproducible output across multiple function calls
        :param max_iter: the maximum number of iterations to perform in the inner optimization loop
        :param generalized: use auxiliary variables (required for >2 views)
        :param initialization: intialization for optimisation. 'unregularized' uses CCA or PLS solution,'random' uses random initialization,'uniform' uses uniform initialization of weights and scores
        :param tol: tolerance value used for early stopping
        """
        super().__init__(latent_dims=latent_dims, scale=scale, centre=centre, copy_data=copy_data, deflation='pls',
                         max_iter=max_iter,
                         generalized=generalized,
                         initialization=initialization, tol=tol, random_state=random_state)

    def _set_loop_params(self):
        self.loop = PLSInnerLoop(max_iter=self.max_iter, generalized=self.generalized,
                                 initialization=self.initialization, tol=self.tol, random_state=self.random_state)


class ElasticCCA(_Iterative):
    """
    Fits an elastic CCA by iterative rescaled elastic net regression

    Citation
    --------
    Waaijenborg, Sandra, Philip C. Verselewel de Witt Hamer, and Aeilko H. Zwinderman. "Quantifying the association between gene expressions and DNA-markers by penalized canonical correlation analysis." Statistical applications in genetics and molecular biology 7.1 (2008).

    :Example:

    >>> from cca_zoo.models import ElasticCCA
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> model = ElasticCCA()
    >>> model.fit(X1,X2)
    """

    def __init__(self, latent_dims: int = 1, scale: bool = True, centre=True, copy_data=True, random_state=None,
                 deflation='cca',
                 max_iter: int = 100,
                 generalized: bool = False,
                 initialization: str = 'unregularized', tol: float = 1e-9,
                 c: Union[Iterable[float], float] = None,
                 l1_ratio: Union[Iterable[float], float] = None,
                 constrained: bool = False, stochastic=False,
                 positive: Union[Iterable[bool], bool] = None):
        """
        Constructor for ElasticCCA

        :param latent_dims: number of latent dimensions to fit
        :param scale: normalize variance in each column before fitting
        :param centre: demean data by column before fitting (and before transforming out of sample
        :param copy_data: If True, X will be copied; else, it may be overwritten
        :param random_state: Pass for reproducible output across multiple function calls
        :param deflation: the type of deflation.
        :param max_iter: the maximum number of iterations to perform in the inner optimization loop
        :param generalized: use auxiliary variables (required for >2 views)
        :param initialization: intialization for optimisation. 'unregularized' uses CCA or PLS solution,'random' uses random initialization,'uniform' uses uniform initialization of weights and scores
        :param tol: tolerance value used for early stopping
        :param c: lasso alpha
        :param l1_ratio: l1 ratio in lasso subproblems
        :param constrained: force unit norm constraint with binary search
        :param stochastic: use stochastic regression optimisers for subproblems
        :param positive: constrain model weights to be positive
        """
        self.c = c
        self.l1_ratio = l1_ratio
        self.constrained = constrained
        self.stochastic = stochastic
        self.positive = positive
        if self.positive is not None and stochastic:
            self.stochastic = False
            warnings.warn(
                'Non negative constraints cannot be used with stochastic regressors. Switching to stochastic=False')
        super().__init__(latent_dims=latent_dims, scale=scale, centre=centre, copy_data=copy_data, deflation=deflation,
                         max_iter=max_iter,
                         generalized=generalized,
                         initialization=initialization, tol=tol, random_state=random_state
                         )

    def _set_loop_params(self):
        self.loop = ElasticInnerLoop(max_iter=self.max_iter, c=self.c, l1_ratio=self.l1_ratio,
                                     generalized=self.generalized, initialization=self.initialization,
                                     tol=self.tol, constrained=self.constrained,
                                     stochastic=self.stochastic, positive=self.positive, random_state=self.random_state)


class CCA_ALS(ElasticCCA):
    """
    Fits a CCA model with CCA deflation by NIPALS algorithm. Implemented by ElasticCCA with 0 regularisation

    Citation
    --------
    Golub, Gene H., and Hongyuan Zha. "The canonical correlations of matrix pairs and their numerical computation." Linear algebra for signal processing. Springer, New York, NY, 1995. 27-49.

    :Example:

    >>> from cca_zoo.models import CCA_ALS
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> model = CCA_ALS()
    >>> model.fit(X1,X2)
    """

    def __init__(self, latent_dims: int = 1, scale: bool = True, centre=True, copy_data=True, random_state=None,
                 max_iter: int = 100,
                 generalized: bool = False,
                 initialization: str = 'random', tol: float = 1e-9, stochastic=True,
                 positive: Union[Iterable[bool], bool] = None):
        """
        Constructor for CCA_ALS

        :param latent_dims: number of latent dimensions to fit
        :param scale: normalize variance in each column before fitting
        :param centre: demean data by column before fitting (and before transforming out of sample
        :param copy_data: If True, X will be copied; else, it may be overwritten
        :param random_state: Pass for reproducible output across multiple function calls
        :param max_iter: the maximum number of iterations to perform in the inner optimization loop
        :param generalized: use auxiliary variables (required for >2 views)
        :param initialization: intialization for optimisation. 'unregularized' uses CCA or PLS solution,'random' uses random initialization,'uniform' uses uniform initialization of weights and scores
        :param tol: tolerance value used for early stopping
        :param stochastic: use stochastic regression optimisers for subproblems
        :param positive: constrain model weights to be positive
        """

        super().__init__(latent_dims=latent_dims, max_iter=max_iter, generalized=generalized,
                         initialization=initialization, tol=tol, constrained=False, stochastic=stochastic,
                         centre=centre, copy_data=copy_data, scale=scale,
                         positive=positive, random_state=random_state, c=1e-3)


class SCCA(ElasticCCA):
    """
    Fits a sparse CCA model by iterative rescaled lasso regression. Implemented by ElasticCCA with l1 ratio=1

    Citation
    --------
    Mai, Qing, and Xin Zhang. "An iterative penalized least squares approach to sparse canonical correlation analysis." Biometrics 75.3 (2019): 734-744.

    :Example:

    >>> from cca_zoo.models import SCCA
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> model = SCCA(c=[0.001,0.001])
    >>> model.fit(X1,X2)
    """

    def __init__(self, latent_dims: int = 1, scale: bool = True, centre=True, copy_data=True, random_state=None,
                 c: Union[Iterable[float], float] = None,
                 max_iter: int = 100,
                 generalized: bool = False,
                 initialization: str = 'unregularized', tol: float = 1e-9, stochastic=False,
                 positive: Union[Iterable[bool], bool] = None):
        """
        Constructor for SCCA

        :param latent_dims: number of latent dimensions to fit
        :param scale: normalize variance in each column before fitting
        :param centre: demean data by column before fitting (and before transforming out of sample
        :param copy_data: If True, X will be copied; else, it may be overwritten
        :param random_state: Pass for reproducible output across multiple function calls
        :param max_iter: the maximum number of iterations to perform in the inner optimization loop
        :param generalized: use auxiliary variables (required for >2 views)
        :param initialization: intialization for optimisation. 'unregularized' uses CCA or PLS solution,'random' uses random initialization,'uniform' uses uniform initialization of weights and scores
        :param tol: tolerance value used for early stopping
        :param c: lasso alpha
        :param stochastic: use stochastic regression optimisers for subproblems
        :param positive: constrain model weights to be positive
        """
        super().__init__(latent_dims=latent_dims, scale=scale, centre=centre, copy_data=copy_data, max_iter=max_iter,
                         generalized=generalized,
                         initialization=initialization, tol=tol, c=c, l1_ratio=1, constrained=False,
                         stochastic=stochastic, positive=positive, random_state=random_state)


class PMD(_Iterative):
    """
    Fits a Sparse CCA (Penalized Matrix Decomposition) model.

    Citation
    --------
    Witten, Daniela M., Robert Tibshirani, and Trevor Hastie. "A penalized matrix decomposition, with applications to sparse principal components and canonical correlation analysis." Biostatistics 10.3 (2009): 515-534.

    :Example:

    >>> from cca_zoo.models import PMD
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> model = PMD(c=[1,1])
    >>> model.fit(X1,X2)
    """

    def __init__(self, latent_dims: int = 1, scale: bool = True, centre=True, copy_data=True, random_state=None,
                 c: Union[Iterable[float], float] = None,
                 max_iter: int = 100,
                 generalized: bool = False, initialization: str = 'unregularized', tol: float = 1e-9,
                 positive: Union[Iterable[bool], bool] = None):
        """
        Constructor for PMD

        :param latent_dims: number of latent dimensions to fit
        :param scale: normalize variance in each column before fitting
        :param centre: demean data by column before fitting (and before transforming out of sample
        :param copy_data: If True, X will be copied; else, it may be overwritten
        :param random_state: Pass for reproducible output across multiple function calls
        :param c: l1 regularisation parameter between 1 and sqrt(number of features) for each view
        :param max_iter: the maximum number of iterations to perform in the inner optimization loop
        :param generalized: use auxiliary variables (required for >2 views)
        :param initialization: intialization for optimisation. 'unregularized' uses CCA or PLS solution,'random' uses random initialization,'uniform' uses uniform initialization of weights and scores
        :param tol: tolerance value used for early stopping
        :param positive: constrain model weights to be positive
        """
        self.c = c
        self.positive = positive
        super().__init__(latent_dims=latent_dims, scale=scale, centre=centre, copy_data=copy_data, max_iter=max_iter,
                         generalized=generalized,
                         initialization=initialization, tol=tol, random_state=random_state)

    def _set_loop_params(self):
        self.loop = PMDInnerLoop(max_iter=self.max_iter, c=self.c, generalized=self.generalized,
                                 initialization=self.initialization, tol=self.tol, positive=self.positive,
                                 random_state=self.random_state)


class ParkhomenkoCCA(_Iterative):
    """
    Fits a sparse CCA (penalized CCA) model

    Citation
    --------
    Parkhomenko, Elena, David Tritchler, and Joseph Beyene. "Sparse canonical correlation analysis with application to genomic data integration." Statistical applications in genetics and molecular biology 8.1 (2009).

    :Example:

    >>> from cca_zoo.models import ParkhomenkoCCA
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> model = ParkhomenkoCCA(c=[0.001,0.001])
    >>> model.fit(X1,X2)
    """

    def __init__(self, latent_dims: int = 1, scale: bool = True, centre=True, copy_data=True, random_state=None,
                 c: Union[Iterable[float], float] = None,
                 max_iter: int = 100,
                 generalized: bool = False, initialization: str = 'unregularized', tol: float = 1e-9):
        """
        Constructor for ParkhomenkoCCA

        :param latent_dims: number of latent dimensions to fit
        :param scale: normalize variance in each column before fitting
        :param centre: demean data by column before fitting (and before transforming out of sample
        :param copy_data: If True, X will be copied; else, it may be overwritten
        :param random_state: Pass for reproducible output across multiple function calls
        :param c: l1 regularisation parameter
        :param max_iter: the maximum number of iterations to perform in the inner optimization loop
        :param generalized: use auxiliary variables (required for >2 views)
        :param initialization: intialization for optimisation. 'unregularized' uses CCA or PLS solution,'random' uses random initialization,'uniform' uses uniform initialization of weights and scores
        :param tol: tolerance value used for early stopping
        """
        self.c = c
        super().__init__(latent_dims=latent_dims, scale=scale, centre=centre, copy_data=copy_data, max_iter=max_iter,
                         generalized=generalized,
                         initialization=initialization, tol=tol, random_state=random_state)

    def _set_loop_params(self):
        self.loop = ParkhomenkoInnerLoop(max_iter=self.max_iter, c=self.c,
                                         generalized=self.generalized,
                                         initialization=self.initialization, tol=self.tol,
                                         random_state=self.random_state)


class SCCA_ADMM(_Iterative):
    """
    Fits a sparse CCA model by alternating ADMM

    Citation
    --------
    Suo, Xiaotong, et al. "Sparse canonical correlation analysis." arXiv preprint arXiv:1705.10865 (2017).

    :Example:

    >>> from cca_zoo.models import SCCA_ADMM
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> model = SCCA_ADMM()
    >>> model.fit(X1,X2)
    """

    def __init__(self, latent_dims: int = 1, scale: bool = True, centre=True, copy_data=True, random_state=None,
                 c: Union[Iterable[float], float] = None,
                 mu: Union[Iterable[float], float] = None,
                 lam: Union[Iterable[float], float] = None,
                 eta: Union[Iterable[float], float] = None,
                 max_iter: int = 100,
                 generalized: bool = False, initialization: str = 'unregularized', tol: float = 1e-9):
        """
        Constructor for SCCA_ADMM

        :param latent_dims: number of latent dimensions to fit
        :param scale: normalize variance in each column before fitting
        :param centre: demean data by column before fitting (and before transforming out of sample
        :param copy_data: If True, X will be copied; else, it may be overwritten
        :param random_state: Pass for reproducible output across multiple function calls
        :param c: l1 regularisation parameter
        :param max_iter: the maximum number of iterations to perform in the inner optimization loop
        :param generalized: use auxiliary variables (required for >2 views)
        :param initialization: intialization for optimisation. 'unregularized' uses CCA or PLS solution,'random' uses random initialization,'uniform' uses uniform initialization of weights and scores
        :param tol: tolerance value used for early stopping
        :param mu:
        :param lam:
        :param: eta:
        """
        self.c = c
        self.mu = mu
        self.lam = lam
        self.eta = eta
        super().__init__(latent_dims=latent_dims, scale=scale, centre=centre, copy_data=copy_data, max_iter=max_iter,
                         generalized=generalized,
                         initialization=initialization, tol=tol, random_state=random_state)

    def _set_loop_params(self):
        self.loop = ADMMInnerLoop(max_iter=self.max_iter, c=self.c, mu=self.mu, lam=self.lam,
                                  eta=self.eta, generalized=self.generalized,
                                  initialization=self.initialization, tol=self.tol, random_state=self.random_state)


class SpanCCA(_Iterative):
    """
    Fits a Sparse CCA model using SpanCCA.

    Citation
    --------
    Asteris, Megasthenis, et al. "A simple and provable algorithm for sparse diagonal CCA." International Conference on Machine Learning. PMLR, 2016.

    """

    def __init__(self, latent_dims: int = 1, scale: bool = True, centre=True, copy_data=True, max_iter: int = 100,
                 generalized: bool = False,
                 initialization: str = 'uniform', tol: float = 1e-9, regularisation='l0',
                 c: Union[Iterable[Union[float, int]], Union[float, int]] = None, rank=1,
                 positive: Union[Iterable[bool], bool] = None, random_state=None):
        """

        :param latent_dims: number of latent dimensions to fit
        :param scale: normalize variance in each column before fitting
        :param centre: demean data by column before fitting (and before transforming out of sample
        :param copy_data: If True, X will be copied; else, it may be overwritten
        :param random_state: Pass for reproducible output across multiple function calls
        :param max_iter: the maximum number of iterations to perform in the inner optimization loop
        :param generalized: use auxiliary variables (required for >2 views)
        :param initialization: intialization for optimisation. 'unregularized' uses CCA or PLS solution,'random' uses random initialization,'uniform' uses uniform initialization of weights and scores
        :param tol: tolerance value used for early stopping
        :param regularisation:
        :param c:
        :param rank:
        :param positive:
        """
        super().__init__(latent_dims=latent_dims, scale=scale, centre=centre, copy_data=copy_data, max_iter=max_iter,
                         generalized=generalized,
                         initialization=initialization, tol=tol, random_state=random_state)
        self.c = c
        self.regularisation = regularisation
        self.rank = rank
        self.positive = positive

    def _set_loop_params(self):
        self.loop = SpanCCAInnerLoop(max_iter=self.max_iter, c=self.c, generalized=self.generalized,
                                     initialization=self.initialization, tol=self.tol,
                                     regularisation=self.regularisation, rank=self.rank, positive=self.positive,
                                     random_state=self.random_state)


class SWCCA(_Iterative):
    """
    A class used to fit SWCCA model

    Citation
    --------
    Wenwen, M. I. N., L. I. U. Juan, and Shihua Zhang. "Sparse Weighted Canonical Correlation Analysis." Chinese Journal of Electronics 27.3 (2018): 459-466.


    """

    def __init__(self, latent_dims: int = 1, scale: bool = True, centre=True, copy_data=True, random_state=None,
                 max_iter: int = 500,
                 generalized: bool = False,
                 initialization: str = 'uniform', tol: float = 1e-9, regularisation='l0',
                 c: Union[Iterable[Union[float, int]], Union[float, int]] = None, sample_support=None,
                 positive: Union[Iterable[bool], bool] = None):
        """

        :param latent_dims: number of latent dimensions to fit
        :param scale: normalize variance in each column before fitting
        :param centre: demean data by column before fitting (and before transforming out of sample
        :param copy_data: If True, X will be copied; else, it may be overwritten
        :param random_state: Pass for reproducible output across multiple function calls
        :param max_iter: the maximum number of iterations to perform in the inner optimization loop
        :param generalized: use auxiliary variables (required for >2 views)
        :param initialization: intialization for optimisation. 'unregularized' uses CCA or PLS solution,'random' uses random initialization,'uniform' uses uniform initialization of weights and scores
        :param tol: tolerance value used for early stopping
        :param regularisation:
        :param c:
        :param sample_support:
        :param positive:
        """

        self.c = c
        self.sample_support = sample_support
        self.regularisation = regularisation
        self.positive = positive
        super().__init__(latent_dims=latent_dims, scale=scale, centre=centre, copy_data=copy_data, max_iter=max_iter,
                         generalized=generalized,
                         initialization=initialization, tol=tol, random_state=random_state)

    def _set_loop_params(self):
        self.loop = SWCCAInnerLoop(max_iter=self.max_iter, generalized=self.generalized,
                                   initialization=self.initialization, tol=self.tol, regularisation=self.regularisation,
                                   c=self.c,
                                   sample_support=self.sample_support, positive=self.positive,
                                   random_state=self.random_state)
