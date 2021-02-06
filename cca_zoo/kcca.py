import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel


# Mostly adapted from:
# https://github.com/lorenzoriano/PyKCCA/blob/master/kcca.py
# With additional inspiration from:
# Copyright (c) 2015, The Regents of the University of California (Regents).
# All rights reserved.

# Permission to use, copy, modify, and distribute this software and its documentation for educational,
# research, and not-for-profit purposes, without fee and without a signed licensing agreement, is hereby granted,
# provided that the above copyright notice, this paragraph and the following two paragraphs appear in all copies,
# modifications, and distributions. Contact The Office of Technology Licensing, UC Berkeley,
# 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201, for commercial licensing opportunities.

# Created by Natalia Bilenko, University of California, Berkeley.

# IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.


class KCCA:
    """
    This is a wrapper class for KCCA solutions

    After initialisation (where the solution is also identified), we can use the method:

    transform(): which allows us to find the latent variable space for out of sample data
    """

    def __init__(self, X: np.array, Y: np.array, latent_dims: int = 2, kernel='linear', sigma=1.0, degree=1, c=None):
        """
        :param X:
        :param Y:
        :param latent_dims: number of latent dimensions to find
        :param kernel: the kernel type 'linear', 'rbf', 'poly'
        :param sigma: sigma parameter used by sklearn rbf kernel
        :param degree: polynomial order parameter used by sklearn polynomial kernel
        :param c: regularisation between 0 (CCA) and 1 (PLS)
        """
        self.X = X
        self.Y = Y
        self.eps = 1e-7
        self.latent_dims = latent_dims
        self.kernel = kernel
        self.sigma = sigma
        self.degree = degree
        self.c = c
        if c is None:
            self.c = [0, 0]
        self.K1 = self.make_kernel(X, X)
        self.K2 = self.make_kernel(Y, Y)
        N = self.K1.shape[0]
        R, D = self.hardoon_method()
        # find what we need to add to D to ensure PSD
        D_smallest_eig = min(0, np.linalg.eigvalsh(D).min()) - self.eps
        betas, alphas = eigh(a=R, b=D - D_smallest_eig * np.eye(D.shape[0]),
                             subset_by_index=[2 * N - latent_dims, 2 * N - 1])
        # sorting according to eigenvalue
        betas = np.real(betas)
        ind = np.argsort(betas)[::-1]
        alphas = alphas[:, ind]
        alpha = alphas[:, :latent_dims]
        # making unit vectors
        alpha = alpha / (np.sum(np.abs(alpha) ** 2, axis=0) ** (1. / 2))
        alpha1 = alpha[:N, :]
        alpha2 = alpha[N:, :]
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
        if self.kernel == 'linear':
            kernel = X @ Y.T
        elif self.kernel == 'rbf':
            kernel = rbf_kernel(X, Y=Y, gamma=(1 / (2 * self.sigma)))
        elif self.kernel == 'poly':
            kernel = polynomial_kernel(X, Y=Y, degree=self.degree)
        else:
            print('invalid kernel: choose linear, rbf, poly')
        return kernel

    def cholesky_method(self):
        self.R1 = np.linalg.cholesky(self.K1)
        self.R2 = np.linalg.cholesky(self.K2)
        self.R1 = (1 - self.c[0]) * self.R1 @ self.R1.T + self.c[0] * np.eye(self.R1.shape[0])
        self.R2 = (1 - self.c[1]) * self.R2 @ self.R2.T + self.c[1] * np.eye(self.R2.shape[0])

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
