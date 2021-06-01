import copy
from abc import abstractmethod
from typing import Union, List

import numpy as np
from sklearn.base import BaseEstimator

from .cca_base import _CCA_Base
from .innerloop import PLSInnerLoop, PMDInnerLoop, ParkhomenkoInnerLoop, ElasticInnerLoop, ADMMInnerLoop


# from hyperopt import fmin, tpe, Trials

class _Iterative(_CCA_Base):
    """
    A class used as the base for iterative CCA methods i.e. those that optimize for each dimension one at a time with deflation.

    """

    def __init__(self, latent_dims: int = 1, deflation='cca', max_iter: int = 100, generalized: bool = False,
                 initialization: str = 'unregularized', tol: float = 1e-9, scale=True):
        """
        Constructor for _Iterative

        :param latent_dims: number of latent dimensions
        :param deflation: the type of deflation.
        :param max_iter: the maximum number of iterations to perform in the inner optimization loop
        :param generalized:
        :param initialization: the initialization for the inner loop either 'unregularized' (initializes with PLS scores and weights)
        or 'random'.
        :param tol: if the cosine similarity of the weights between subsequent iterations is greater than 1-tol the loop is considered converged
        """
        super().__init__(latent_dims=latent_dims, scale=scale)
        self.max_iter = max_iter
        self.generalized = generalized
        self.initialization = initialization
        self.tol = tol
        self.deflation = deflation

    def fit(self, *views: np.ndarray, ):
        """
        Fits the model by running an inner loop to convergence and then using deflation (currently only supports CCA deflation)

        :param views: numpy arrays with the same number of rows (samples) separated by commas
        """
        self._set_loop_params()
        train_views = self.centre_scale(*views)
        n = train_views[0].shape[0]
        p = [view.shape[1] for view in train_views]
        # list of d: p x k
        self.weights_list = [np.zeros((p_, self.latent_dims)) for p_ in p]
        self.loading_list = [np.zeros((p_, self.latent_dims)) for p_ in p]

        # list of d: n x k
        self.score_list = [np.zeros((n, self.latent_dims)) for _ in train_views]

        residuals = copy.deepcopy(list(train_views))

        self.objective = []
        # For each of the dimensions
        for k in range(self.latent_dims):
            self.loop = self.loop.fit(*residuals)
            for i, residual in enumerate(residuals):
                self.weights_list[i][:, k] = self.loop.weights[i]
                self.score_list[i][:, k] = self.loop.scores[i]
                self.loading_list[i][:, k] = np.dot(self.loop.scores[i], residual)
                # TODO This is CCA deflation (https://ars.els-cdn.com/content/image/1-s2.0-S0006322319319183-mmc1.pdf)
                residuals[i] = self.deflate(residuals[i], self.score_list[i][:, k])
            self.objective.append(self.loop.track_objective)
        self.train_correlations = self.predict_corr(*views)
        return self

    def deflate(self, residual, score):
        """
        Deflate view residual by CCA deflation (https://ars.els-cdn.com/content/image/1-s2.0-S0006322319319183-mmc1.pdf)

        :param residual:
        :param score:

        """
        if self.deflation == 'cca':
            return residual - np.outer(score, score) @ residual / np.dot(score, score).item()

    @abstractmethod
    def _set_loop_params(self):
        """
        Sets up the inner optimization loop for the method. By default uses the PLS inner loop.
        """
        self.loop = PLSInnerLoop(max_iter=self.max_iter, generalized=self.generalized,
                                 initialization=self.initialization)


class PLS(_Iterative):
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

    def __init__(self, latent_dims: int = 1, max_iter: int = 100, generalized: bool = False,
                 initialization: str = 'unregularized', tol: float = 1e-9, scale=True):
        """
        Constructor for PLS

        :param latent_dims: number of latent dimensions
        :param max_iter: the maximum number of iterations to perform in the inner optimization loop
        :param generalized:
        :param initialization: the initialization for the inner loop either 'unregularized' (initializes with PLS scores and weights) or 'random'.
        :param tol: if the cosine similarity of the weights between subsequent iterations is greater than 1-tol the loop is considered converged
        """
        super().__init__(latent_dims=latent_dims, max_iter=max_iter, generalized=generalized,
                         initialization=initialization, tol=tol, scale=scale)

    def _set_loop_params(self):
        self.loop = PLSInnerLoop(max_iter=self.max_iter, generalized=self.generalized,
                                 initialization=self.initialization, tol=self.tol)


class ElasticCCA(_Iterative, BaseEstimator):
    """
    Fits an elastic CCA by iterative rescaled elastic net regression

    :Example:

    >>> from cca_zoo.models import ElasticCCA
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> model = ElasticCCA()
    >>> model.fit(X1,X2)
    """

    def __init__(self, latent_dims: int = 1, c: Union[List[float], float] = None,
                 l1_ratio: Union[List[float], float] = None,
                 constrained: bool = False, max_iter: int = 100,
                 generalized: bool = False,
                 initialization: str = 'unregularized', tol: float = 1e-9, stochastic=False, scale=True):
        """
        Constructor for ElasticCCA

        :param latent_dims: Number of latent dimensions
        :param c: lasso alpha
        :param l1_ratio: l1 ratio in lasso subproblems
        :param max_iter: Maximum number of iterations
        """
        self.c = c
        self.l1_ratio = l1_ratio
        self.constrained = constrained
        self.stochastic = stochastic
        super().__init__(latent_dims=latent_dims, max_iter=max_iter, generalized=generalized,
                         initialization=initialization, tol=tol, scale=scale)

    def _set_loop_params(self):
        self.loop = ElasticInnerLoop(max_iter=self.max_iter, c=self.c, l1_ratio=self.l1_ratio,
                                     generalized=self.generalized, initialization=self.initialization,
                                     tol=self.tol, constrained=self.constrained,
                                     stochastic=self.stochastic)


class CCA_ALS(ElasticCCA):
    """
    Fits a CCA model with CCA deflation by NIPALS algorithm. Implemented by ElasticCCA with 0 regularisation

    :Example:

    >>> from cca_zoo.models import CCA_ALS
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> model = CCA_ALS()
    >>> model.fit(X1,X2)
    """

    def __init__(self, latent_dims: int = 1, max_iter: int = 100, generalized: bool = False,
                 initialization: str = 'random', tol: float = 1e-9, stochastic=True, scale=True):
        """
        Constructor for CCA_ALS

        :param latent_dims: number of latent dimensions
        :param max_iter: the maximum number of iterations to perform in the inner optimization loop
        :param generalized:
        :param initialization: the initialization for the inner loop either 'unregularized' (initializes with PLS scores and weights) or 'random'.
        :param tol: if the cosine similarity of the weights between subsequent iterations is greater than 1-tol the loop is considered converged
        """
        super().__init__(latent_dims=latent_dims, max_iter=max_iter, generalized=generalized,
                         initialization=initialization, tol=tol, constrained=False, stochastic=stochastic, scale=scale)


class SCCA(ElasticCCA):
    """
    Fits a sparse CCA model by iterative rescaled lasso regression. Implemented by ElasticCCA with l1 ratio=1

    :Example:

    >>> from cca_zoo.models import SCCA
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> model = SCCA(c=[0.001,0.001])
    >>> model.fit(X1,X2)
    """

    def __init__(self, latent_dims: int = 1, c: List[float] = None, max_iter: int = 100,
                 generalized: bool = False,
                 initialization: str = 'unregularized', tol: float = 1e-9, stochastic=False, scale=True):
        """
        Constructor for SCCA

        :param latent_dims: number of latent dimensions
        :param c: l1 regularisation parameter
        :param max_iter: the maximum number of iterations to perform in the inner optimization loop
        :param generalized:
        :param initialization: the initialization for the inner loop either 'unregularized' (initializes with PLS scores and weights) or 'random'.
        :param tol: if the cosine similarity of the weights between subsequent iterations is greater than 1-tol the loop is considered converged
        """
        super().__init__(latent_dims=latent_dims, max_iter=max_iter, generalized=generalized,
                         initialization=initialization, tol=tol, c=c, l1_ratio=1, constrained=False,
                         stochastic=stochastic, scale=scale)


class PMD(_Iterative, BaseEstimator):
    """
    Fits a Sparse CCA (Penalized Matrix Decomposition) model.

    :Example:

    >>> from cca_zoo.models import PMD
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> model = PMD(c=[1,1])
    >>> model.fit(X1,X2)
    """

    def __init__(self, latent_dims: int = 1, c: List[float] = None, max_iter: int = 100,
                 generalized: bool = False, initialization: str = 'unregularized', tol: float = 1e-9, scale=True):
        """
        Constructor for PMD

        :param latent_dims: number of latent dimensions
        :param c: l1 regularisation parameter between 1 and sqrt(number of features) for each view
        :param max_iter: the maximum number of iterations to perform in the inner optimization loop
        :param generalized:
        :param initialization: the initialization for the inner loop either 'unregularized' (initializes with PLS scores and weights) or 'random'.
        :param tol: if the cosine similarity of the weights between subsequent iterations is greater than 1-tol the loop is considered converged
        """
        self.c = c
        super().__init__(latent_dims=latent_dims, max_iter=max_iter, generalized=generalized,
                         initialization=initialization, tol=tol, scale=scale)

    def _set_loop_params(self):
        self.loop = PMDInnerLoop(max_iter=self.max_iter, c=self.c, generalized=self.generalized,
                                 initialization=self.initialization, tol=self.tol)


class ParkhomenkoCCA(_Iterative, BaseEstimator):
    """
    Fits a sparse CCA (penalized CCA) model

    :Example:

    >>> from cca_zoo.models import ParkhomenkoCCA
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> model = ParkhomenkoCCA(c=[0.001,0.001])
    >>> model.fit(X1,X2)
    """

    def __init__(self, latent_dims: int = 1, c: List[float] = None, max_iter: int = 100,
                 generalized: bool = False, initialization: str = 'unregularized', tol: float = 1e-9, scale=True):
        """
        Constructor for ParkhomenkoCCA

        :param latent_dims: number of latent dimensions
        :param c: l1 regularisation parameter
        :param max_iter: the maximum number of iterations to perform in the inner optimization loop
        :param generalized:
        :param initialization: the initialization for the inner loop either 'unregularized' (initializes with PLS scores and weights) or 'random'.
        :param tol: if the cosine similarity of the weights between subsequent iterations is greater than 1-tol the loop is considered converged
        """
        self.c = c
        super().__init__(latent_dims=latent_dims, max_iter=max_iter, generalized=generalized,
                         initialization=initialization, tol=tol, scale=scale)

    def _set_loop_params(self):
        self.loop = ParkhomenkoInnerLoop(max_iter=self.max_iter, c=self.c,
                                         generalized=self.generalized,
                                         initialization=self.initialization, tol=self.tol)


class SCCA_ADMM(_Iterative, BaseEstimator):
    """
    Fits a sparse CCA model by alternating ADMM

    :Example:

    >>> from cca_zoo.models import SCCA_ADMM
    >>> X1 = np.random.rand(10,5)
    >>> X2 = np.random.rand(10,5)
    >>> model = SCCA_ADMM()
    >>> model.fit(X1,X2)
    """

    def __init__(self, latent_dims: int = 1, c: List[float] = None, mu: List[float] = None, lam: List[float] = None,
                 eta: List[float] = None,
                 max_iter: int = 100,
                 generalized: bool = False, initialization: str = 'unregularized', tol: float = 1e-9, scale=True):
        """
        Constructor for SCCA_ADMM

        :param latent_dims: number of latent dimensions
        :param c: l1 regularisation parameter
        :param mu:
        :param lam:
        :param: eta:
        :param max_iter: the maximum number of iterations to perform in the inner optimization loop
        :param generalized:
        :param initialization: the initialization for the inner loop either 'unregularized' (initializes with PLS scores and weights) or 'random'.
        :param tol: if the cosine similarity of the weights between subsequent iterations is greater than 1-tol the loop is considered converged
        """
        self.c = c
        self.mu = mu
        self.lam = lam
        self.eta = eta
        super().__init__(latent_dims=latent_dims, max_iter=max_iter, generalized=generalized,
                         initialization=initialization, tol=tol, scale=scale)

    def _set_loop_params(self):
        self.loop = ADMMInnerLoop(max_iter=self.max_iter, c=self.c, mu=self.mu, lam=self.lam,
                                  eta=self.eta, generalized=self.generalized,
                                  initialization=self.initialization, tol=self.tol)
