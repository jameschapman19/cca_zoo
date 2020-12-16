import numpy as np
from scipy.linalg import eigh
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel


# https://github.com/lorenzoriano/PyKCCA/blob/master/kcca.py
# Also some adaptation of pyrcca package.

class KCCA:
    """
    This is a wrapper class for KCCA solutions

    After initialisation (where the solution is also identified), we can use the method:

    transform(): which allows us to find the latent variable space for out of sample data
    """

    def __init__(self, X: np.array, Y: np.array, params: dict = None, latent_dims: int = 2):
        """
        :param X:
        :param Y:
        :param params: a dictionary containing the relevant parameters required for the model. If None use defaults
        :param latent_dims: number of latent dimensions to find
        """
        self.X = X
        self.Y = Y
        self.eps = 1e-9
        self.latent_dims = latent_dims
        self.ktype = params.get('kernel')
        self.sigma = params.get('sigma')
        self.degree = params.get('degree')
        self.c = params.get('c')
        self.K1 = self.make_kernel(X, X)
        self.K2 = self.make_kernel(Y, Y)

        N = self.K1.shape[0]

        R, D = self.hardoon_method()
        betas, alphas = eigh(R, D + self.eps * np.eye(D.shape[0]))
        # sorting according to eigenvalue
        betas = np.real(betas)
        ind = np.argsort(betas)

        alphas = alphas[:, ind]
        alpha = alphas[:, :latent_dims]
        # making unit vectors
        alpha = alpha / (np.sum(np.abs(alpha) ** 2, axis=0) ** (1. / 2))
        alpha1 = alpha[:N, :]
        alpha2 = -alpha[N:, :]
        self.U = np.dot(self.K1, alpha1).T
        self.V = np.dot(self.K2, alpha2).T
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def make_kernel(self, X: np.array, Y: np.array):
        """
        :param X:
        :param Y:
        :return: Kernel matrix
        """
        if self.ktype == 'linear':
            kernel = X @ Y.T
        elif self.ktype == 'rbf':
            kernel = rbf_kernel(X, Y=Y, gamma=(1 / (2 * self.sigma)))
        elif self.ktype == 'poly':
            kernel = polynomial_kernel(X, Y=Y, degree=self.degree)
        else:
            print('invalid kernel: choose linear, rbf, poly')
        return kernel

    def hardoon_method(self):
        N = self.K1.shape[0]
        Z = np.zeros((N, N))

        R1 = np.c_[Z, np.dot(self.K1, self.K2)]
        R2 = np.c_[np.dot(self.K2, self.K1), Z]
        R = np.r_[R1, R2]

        D1 = np.c_[(1 - self.c[0]) * self.K1 @ self.K1.T + self.c[0] * self.K1, Z]
        D2 = np.c_[Z, (1 - self.c[1]) * self.K2 @ self.K2.T + self.c[1] * self.K2]
        D = np.r_[D1, D2]
        return R, D

    def transform(self, X_test: np.array = None, Y_test: np.array = None):
        """
        :param X_test:
        :param Y_test:
        :return: Test data transformed into kernel feature space
        """
        if X_test is not None:
            Ktest = self.make_kernel(X_test, self.X)
            U_test = np.dot(Ktest, self.alpha1)
        if Y_test is not None:
            Ktest = self.make_kernel(Y_test, self.Y)
            V_test = np.dot(Ktest, self.alpha2)
        return U_test, V_test
