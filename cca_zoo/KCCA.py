import numpy as np
from scipy.linalg import eigh
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel


# https://github.com/lorenzoriano/PyKCCA/blob/master/kcca.py

class KCCA:
    """
    This is a wrapper class for KCCA solutions

    After initialisation (where the solution is also identified), we can use the method:

    transform(): which allows us to find the latent variable space for out of sample data
    """

    def __init__(self, X: np.array, Y: np.array, params: dict = None, latent_dims: int = 2):
        self.X = X
        self.Y = Y
        self.eps = 1e-10
        self.latent_dims = latent_dims
        self.ktype = params.get('kernel')
        self.sigma = params.get('sigma')
        self.degree = params.get('degree')
        self.c = params.get('c')
        self.K1 = self.make_kernel(X, X)
        self.K2 = self.make_kernel(Y, Y)

        N = self.K1.shape[0]
        N0 = np.eye(N) - 1. / N * np.ones(N)
        self.K1 = np.dot(np.dot(N0, self.K1), N0)
        self.K2 = np.dot(np.dot(N0, self.K2), N0)

        R, D = self.hardoon_method()
        betas, alphas = eigh(R, D)
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

    def make_kernel(self, X: np.array, Y: np.array, normalize: bool = True):
        if self.ktype == 'linear':
            kernel = linear_kernel(X, Y=Y)
        elif self.ktype == 'rbf':
            kernel = rbf_kernel(X, Y=Y, gamma=(1 / (2 * self.sigma)))
        elif self.ktype == 'poly':
            kernel = polynomial_kernel(X, Y=Y, degree=self.degree)
        if normalize:
            kernel = kernel / np.linalg.svd(kernel)[1].max()
        return kernel

    def hardoon_method(self):
        N = self.K1.shape[0]
        Z = np.zeros((N, N))

        R1 = np.c_[Z, np.dot(self.K1, self.K2)]
        R2 = np.c_[np.dot(self.K2, self.K1), Z]
        R = np.r_[R1, R2]

        D1 = np.c_[self.K1 @ self.K1.T + self.c[0] * np.eye(self.K1.shape[0]), Z]
        D2 = np.c_[Z, self.K2 @ self.K2.T + self.c[1] * np.eye(self.K2.shape[0])]
        D = np.r_[D1, D2]
        return R, D

    def transform(self, X_test: np.array = None, Y_test: np.array = None):
        n_dims = self.alpha1.shape[1]
        if X_test is not None:
            Ktest = self.make_kernel(X_test, self.X)
            U_test = np.dot(Ktest, self.alpha1[:, :n_dims])
        if Y_test is not None:
            Ktest = self.make_kernel(Y_test, self.Y)
            V_test = np.dot(Ktest, self.alpha2[:, :n_dims])
        return U_test, V_test
