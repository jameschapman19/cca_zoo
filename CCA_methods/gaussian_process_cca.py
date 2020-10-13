import os

import numpy as np
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import pyro.ops.stats as stats
import torch
from pyro.contrib.gp.parameterized import Parameterized
from pyro.contrib.gp.util import conditional
from pyro.nn.module import PyroParam
from pyro.nn.module import PyroParam, pyro_method
from torch.distributions import constraints
from torch.nn import Parameter

from CCA_methods import linear
from CCA_methods.generate_data import generate_candola

smoke_test = ('CI' in os.environ)  # ignore; used to check code integrity in the Pyro repo
pyro.enable_validation(True)  # can help with debugging


class Wrapper:
    """
    This is a wrapper class for linear, regularised and kernel  CCA, Multiset CCA and Generalized CCA.
    We create an instance with a method and number of latent dimensions.
    If we have more than 2 views we need to use generalized methods, but we can override in the 2 view case also with
    the generalized parameter.

    The class has a number of methods:

    fit(): gives us train correlations and stores the variables needed for out of sample prediction as well as some
    method-specific variables

    predict_corr(): allows us to predict the out of sample correlation for supplied views
    """

    def __init__(self, outdim_size=2, sparse_x=True, sparse_y=True, noise=1e-3, jitter=1e-3, steps_gpcca=500,
                 steps_gp_y=500,
                 steps_gp_z=500, tol=1e-5):
        self.outdim_size = outdim_size
        self.jitter = jitter
        self.sparse_x = sparse_x
        self.sparse_y = sparse_y
        self.steps_gpcca = steps_gpcca
        self.steps_gp_y = steps_gp_y
        self.steps_gp_z = steps_gp_z
        self.noise = noise
        self.tol = tol

    def fit(self, X_train, Y_train):
        z_data = torch.tensor(X_train, dtype=torch.float64)
        # we need to transpose data to correct its shape
        z_data = z_data.t()

        y_data = torch.tensor(Y_train, dtype=torch.float64)
        # we need to transpose data to correct its shape
        y_data = y_data.t()

        # we setup the mean of our prior over X
        # shape: 437 x 2
        warm_start_cca = linear.Wrapper(outdim_size=self.outdim_size).fit(X_train, Y_train)

        X_prior_mean = torch.tensor(((warm_start_cca.U + warm_start_cca.V) / 2).T, dtype=torch.float64)

        kernel = gp.kernels.RBF(input_dim=self.outdim_size, lengthscale=torch.ones(self.outdim_size))

        # we clone here so that we don't change our prior during the course of training
        X = Parameter(X_prior_mean.clone())

        self.gpcca = GPCCA(X, y_data, z_data, kernel, noise=torch.tensor(self.noise), jitter=self.jitter)

        # we use `.to_event()` to tell Pyro that the prior distribution for X has no batch_shape
        self.gpcca.X = pyro.nn.PyroSample(dist.Normal(X_prior_mean, 0.1).to_event())
        self.gpcca.autoguide("X", dist.Normal)

        self.gpcca_losses = gp.util.train(self.gpcca, num_steps=self.steps_gpcca)

        self.gpcca.mode = "guide"
        X = self.gpcca.X  # draw a sample from the guide of the variable X

        if self.sparse_x:
            Xu = stats.resample(X.clone(), 32)
            self.gp_z = gp.models.SparseGPRegression(z_data.t(), X.detach().t(), kernel, Xu=Xu,
                                                     noise=torch.tensor(self.noise), jitter=1e-3)
        else:
            self.gp_z = gp.models.GPRegression(z_data.t(), X.detach().t(), kernel, noise=torch.tensor(self.noise),
                                               jitter=1e-3)

        if self.sparse_y:
            Xu = stats.resample(X.clone(), 32)
            self.gp_y = gp.models.SparseGPRegression(y_data.t(), X.detach().t(), kernel, Xu=Xu,
                                                     noise=torch.tensor(self.noise), jitter=1e-3)
        else:
            self.gp_y = gp.models.GPRegression(y_data.t(), X.detach().t(), kernel, noise=torch.tensor(self.noise),
                                               jitter=1e-3)

        self.gp_z_losses = gp.util.train(self.gp_z, num_steps=self.steps_gp_z)

        self.gp_y_losses = gp.util.train(self.gp_y, num_steps=self.steps_gp_y)

        self.gp_z.mode = "guide"
        self.U = self.gp_z(z_data.t())[0].detach().numpy()  # draw a sample from the guide of the variable X
        self.U_err = self.gp_z(z_data.t())[1].detach().numpy()

        self.gp_y.mode = "guide"
        self.V = self.gp_y(y_data.t())[0].detach().numpy()  # draw a sample from the guide of the variable X
        self.V_err = self.gp_y(y_data.t())[1].detach().numpy()  # draw a sample from the guide of the variable X

        self.U /= np.linalg.norm(self.U, axis=1, keepdims=True)
        self.V /= np.linalg.norm(self.V, axis=1, keepdims=True)
        self.train_correlations = np.diag(np.corrcoef(self.U, self.V)[:self.outdim_size, self.outdim_size:])
        return self

    def predict_corr(self, X_test, Y_test):
        z_test_data = torch.tensor(X_test, dtype=torch.float64)
        # we need to transpose data to correct its shape
        z_test_data = z_test_data.t()

        y_test_data = torch.tensor(Y_test, dtype=torch.float64)
        # we need to transpose data to correct its shape
        y_test_data = y_test_data.t()

        V_test = self.gp_y(y_test_data.t())[0].detach().numpy()
        V_test_err = self.gp_y(y_test_data.t())[1].detach().numpy()

        U_test = self.gp_z(z_test_data.t())[0].detach().numpy()  # draw a sample from the guide of the variable X
        U_test_err = self.gp_z(z_test_data.t())[1].detach().numpy()
        #
        U_test /= np.linalg.norm(U_test, axis=1, keepdims=True)
        V_test /= np.linalg.norm(V_test, axis=1, keepdims=True)
        correlations = np.diag(np.corrcoef(U_test, V_test)[:self.outdim_size, self.outdim_size:])

        """
        plt.figure()
        plt.scatter(U_test[0], V_test[0])
        plt.figure()
        plt.scatter(U_test[1], V_test[1])
        """
        return correlations


def _zero_mean_function(x):
    return 0


class GPCCA(Parameterized):
    r"""
    Gaussian Process Regression model.

    The core of a Gaussian Process is a covariance function :math:`k` which governs
    the similarity between input points. Given :math:`k`, we can establish a
    distribution over functions :math:`f` by a multivarite normal distribution

    .. math:: p(f(X)) = \mathcal{N}(0, k(X, X)),

    where :math:`X` is any set of input points and :math:`k(X, X)` is a covariance
    matrix whose entries are outputs :math:`k(x, z)` of :math:`k` over input pairs
    :math:`(x, z)`. This distribution is usually denoted by

    .. math:: f \sim \mathcal{GP}(0, k).

    .. note:: Generally, beside a covariance matrix :math:`k`, a Gaussian Process can
        also be specified by a mean function :math:`m` (which is a zero-value function
        by default). In that case, its distribution will be

        .. math:: p(f(X)) = \mathcal{N}(m(X), k(X, X)).

    Given inputs :math:`X` and their noisy observations :math:`y`, the Gaussian Process
    Regression model takes the form

    .. math::
        f &\sim \mathcal{GP}(0, k(X, X)),\\
        y & \sim f + \epsilon,

    where :math:`\epsilon` is Gaussian noise.

    .. note:: This model has :math:`\mathcal{O}(N^3)` complexity for training,
        :math:`\mathcal{O}(N^3)` complexity for testing. Here, :math:`N` is the number
        of train inputs.

    Reference:

    [1] `Gaussian Processes for Machine Learning`,
    Carl E. Rasmussen, Christopher K. I. Williams

    :param torch.Tensor X: A input data for training. Its first dimension is the number
        of data points.
    :param torch.Tensor y: An output data for training. Its last dimension is the
        number of data points.
    :param ~pyro.contrib.gp.kernels.kernel.Kernel kernel: A Pyro kernel object, which
        is the covariance function :math:`k`.
    :param torch.Tensor noise: Variance of Gaussian noise of this model.
    :param callable mean_function: An optional mean function :math:`m` of this Gaussian
        process. By default, we use zero mean.
    :param float jitter: A small positive term which is added into the diagonal part of
        a covariance matrix to help stablize its Cholesky decomposition.
    :param str name: Name of this model.
    """

    def __init__(self, X, y_1, y_2, kernel, Xu=None, noise=None, mean_function=None, jitter=1e-6, approx=None,
                 name="GPCCA"):
        super(GPCCA, self).__init__()
        self.sparse = False
        if Xu is not None:
            self.sparse = True
            self.Xu = Xu

        noise = self.X.new_ones(()) if noise is None else noise

        noise = self.X.new_tensor(1.) if noise is None else noise
        self.noise = PyroParam(noise, constraints.positive)
        self.set_data(X, y_1, y_2)
        self.kernel = kernel
        self.mean_function = (mean_function if mean_function is not None else
                              _zero_mean_function)
        self.jitter = jitter

        if approx is None:
            self.approx = "VFE"
        elif approx in ["DTC", "FITC", "VFE"]:
            self.approx = approx
        else:
            raise ValueError("The sparse approximation method should be one of "
                             "'DTC', 'FITC', 'VFE'.")

    @pyro_method
    def model(self):
        self.set_mode("model")

        N = self.X.size(0)
        zero_loc = self.X.new_zeros(N)
        f_loc_1 = zero_loc + self.mean_function(self.X)
        f_loc_2 = zero_loc + self.mean_function(self.X)
        if self.sparse:
            M = self.Xu.size(0)
            Kuu = self.kernel(self.Xu).contiguous()
            Kuu.view(-1)[::M + 1] += self.jitter  # add jitter to the diagonal
            Luu = Kuu.cholesky()
            Kuf = self.kernel(self.Xu, self.X)
            W = Kuf.triangular_solve(Luu, upper=False)[0].t()
            D = self.noise.expand(N)
            if self.approx == "FITC" or self.approx == "VFE":
                Kffdiag = self.kernel(self.X, diag=True)
                Qffdiag = W.pow(2).sum(dim=-1)
                if self.approx == "FITC":
                    D = D + Kffdiag - Qffdiag
                else:  # approx = "VFE"
                    trace_term = (Kffdiag - Qffdiag).sum() / self.noise
                    trace_term = trace_term.clamp(min=0)
            if self.y_1 is None or self.y_2 is None:
                f_var_1 = D + W.pow(2).sum(dim=-1)
                f_var_2 = D + W.pow(2).sum(dim=-1)
                return f_loc_1, f_var_1, f_loc_2, f_var_2
            else:
                if self.approx == "VFE":
                    pyro.factor(self._pyro_get_fullname("trace_term"), -trace_term / 2.)

                return pyro.sample(self._pyro_get_fullname("y_1"),
                                   dist.LowRankMultivariateNormal(f_loc_1, W, D)
                                   .expand_by(self.y_1.shape[:-1])
                                   .to_event(self.y_1.dim() - 1),
                                   obs=self.y_1), pyro.sample(self._pyro_get_fullname("y_2"),
                                                              dist.LowRankMultivariateNormal(f_loc_2, W, D)
                                                              .expand_by(self.y_2.shape[:-1])
                                                              .to_event(self.y_2.dim() - 1),
                                                              obs=self.y_2)

        else:
            Kff = self.kernel(self.X)
            Kff.view(-1)[::N + 1] += self.jitter + self.noise  # add noise to diagonal
            Lff = Kff.cholesky()

            if self.y_1 is None or self.y_2 is None:
                f_var_1 = Lff.pow(2).sum(dim=-1)
                f_var_2 = Lff.pow(2).sum(dim=-1)
                return f_loc_1, f_var_1, f_loc_2, f_var_2
            else:
                return pyro.sample(self._pyro_get_fullname("y_1"),
                                   dist.MultivariateNormal(f_loc_1, scale_tril=Lff)
                                   .expand_by(self.y_1.shape[:-1])
                                   .to_event(self.y_1.dim() - 1),
                                   obs=self.y_1), pyro.sample(self._pyro_get_fullname("y_2"),
                                                              dist.MultivariateNormal(f_loc_2, scale_tril=Lff)
                                                              .expand_by(self.y_2.shape[:-1])
                                                              .to_event(self.y_2.dim() - 1),
                                                              obs=self.y_2)

    @pyro_method
    def guide(self):
        self.set_mode("guide")
        self._load_pyro_samples()

    def forward(self, Xnew, full_cov=False, noiseless=True):
        r"""
        Computes the mean and covariance matrix (or variance) of Gaussian Process
        posterior on a test input data :math:`X_{new}`:
        .. math:: p(f^* \mid X_{new}, X, y, k, \epsilon) = \mathcal{N}(loc, cov).
        .. note:: The noise parameter ``noise`` (:math:`\epsilon`) together with
            kernel's parameters have been learned from a training procedure (MCMC or
            SVI).
        :param torch.Tensor Xnew: A input data for testing. Note that
            ``Xnew.shape[1:]`` must be the same as ``self.X.shape[1:]``.
        :param bool full_cov: A flag to decide if we want to predict full covariance
            matrix or just variance.
        :param bool noiseless: A flag to decide if we want to include noise in the
            prediction output or not.
        :returns: loc and covariance matrix (or variance) of :math:`p(f^*(X_{new}))`
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        self._check_Xnew_shape(Xnew)
        self.set_mode("guide")
        N = self.X.size(0)

        if self.sparse:
            M = self.Xu.size(0)

            # TODO: cache these calculations to get faster inference

            Kuu = self.kernel(self.Xu).contiguous()
            Kuu.view(-1)[::M + 1] += self.jitter  # add jitter to the diagonal
            Luu = Kuu.cholesky()

            Kuf = self.kernel(self.Xu, self.X)

            W = Kuf.triangular_solve(Luu, upper=False)[0]
            D = self.noise.expand(N)
            if self.approx == "FITC":
                Kffdiag = self.kernel(self.X, diag=True)
                Qffdiag = W.pow(2).sum(dim=0)
                D = D + Kffdiag - Qffdiag

            W_Dinv = W / D
            K = W_Dinv.matmul(W.t()).contiguous()
            K.view(-1)[::M + 1] += 1  # add identity matrix to K
            L = K.cholesky()

            # get y_residual and convert it into 2D tensor for packing
            y_1_residual = self.y_1 - self.mean_function(self.X)
            y_1_2D = y_1_residual.reshape(-1, N).t()
            W_Dinv_y_1 = W_Dinv.matmul(y_1_2D)
            y_2_residual = self.y_2 - self.mean_function(self.X)
            y_2_2D = y_2_residual.reshape(-1, N).t()
            W_Dinv_y_2 = W_Dinv.matmul(y_2_2D)

            # End caching ----------

            Kus = self.kernel(self.Xu, Xnew)
            Ws = Kus.triangular_solve(Luu, upper=False)[0]
            pack_1 = torch.cat((W_Dinv_y_1, Ws), dim=1)
            Linv_pack_1 = pack_1.triangular_solve(L, upper=False)[0]
            pack_2 = torch.cat((W_Dinv_y_2, Ws), dim=1)
            Linv_pack_2 = pack_2.triangular_solve(L, upper=False)[0]
            # unpack
            Linv_W_Dinv_y_1 = Linv_pack_1[:, :W_Dinv_y_1.shape[1]]
            Linv_Ws_1 = Linv_pack_1[:, W_Dinv_y_1.shape[1]:]
            Linv_W_Dinv_y_2 = Linv_pack_2[:, :W_Dinv_y_2.shape[1]]
            Linv_Ws_2 = Linv_pack_2[:, W_Dinv_y_2.shape[1]:]

            C = Xnew.size(0)
            loc_1_shape = self.y_1.shape[:-1] + (C,)
            loc_1 = Linv_W_Dinv_y_1.t().matmul(Linv_Ws_1).reshape(loc_1_shape)
            loc_2_shape = self.y_2.shape[:-1] + (C,)
            loc_2 = Linv_W_Dinv_y_2.t().matmul(Linv_Ws_2).reshape(loc_2_shape)

            if full_cov:
                Kss = self.kernel(Xnew).contiguous()
                if not noiseless:
                    Kss.view(-1)[::C + 1] += self.noise  # add noise to the diagonal
                Qss = Ws.t().matmul(Ws)
                cov_1 = Kss - Qss + Linv_Ws_1.t().matmul(Linv_Ws_1)
                cov_1_shape = self.y_1.shape[:-1] + (C, C)
                cov_1 = cov_1.expand(cov_1_shape)
                cov_2 = Kss - Qss + Linv_Ws_2.t().matmul(Linv_Ws_1)
                cov_2_shape = self.y_2.shape[:-1] + (C, C)
                cov_2 = cov_2.expand(cov_2_shape)
            else:
                Kssdiag = self.kernel(Xnew, diag=True)
                if not noiseless:
                    Kssdiag = Kssdiag + self.noise
                Qssdiag = Ws.pow(2).sum(dim=0)
                cov_1 = Kssdiag - Qssdiag + Linv_Ws_1.pow(2).sum(dim=0)
                cov_1_shape = self.y_1.shape[:-1] + (C,)
                cov_1 = cov_1.expand(cov_1_shape)
                cov_2 = Kssdiag - Qssdiag + Linv_Ws_2.pow(2).sum(dim=0)
                cov_2_shape = self.y_2.shape[:-1] + (C,)
                cov_2 = cov_2.expand(cov_2_shape)

            return loc_1 + self.mean_function(Xnew), cov_1, loc_2 + self.mean_function(Xnew), cov_2

        else:
            Kff = self.kernel(self.X).contiguous()
            Kff.view(-1)[::N + 1] += self.jitter + self.noise  # add noise to the diagonal
            Lff = Kff.cholesky()

            y_1_residual = self.y_1 - self.mean_function(self.X)
            y_2_residual = self.y_2 - self.mean_function(self.X)
            loc_1, cov_1 = conditional(Xnew, self.X, self.kernel, y_1_residual, None, Lff,
                                       full_cov, jitter=self.jitter)
            loc_2, cov_2 = conditional(Xnew, self.X, self.kernel, y_2_residual, None, Lff,
                                       full_cov, jitter=self.jitter)

            if full_cov and not noiseless:
                M = Xnew.size(0)
                cov_1 = cov_1.contiguous()
                cov_1.view(-1, M * M)[:, ::M + 1] += self.noise  # add noise to the diagonal

                M = Xnew.size(0)
                cov_2 = cov_2.contiguous()
                cov_2.view(-1, M * M)[:, ::M + 1] += self.noise  # add noise to the diagonal
            if not full_cov and not noiseless:
                cov_1 = cov_1 + self.noise
                cov_2 = cov_2 + self.noise

            return loc_1 + self.mean_function(Xnew), cov_1, loc_2 + self.mean_function(Xnew), cov_2

    def set_data(self, X, y_1=None, y_2=None):
        """
        Sets data for Gaussian Process models.

        References:

        [1] `Scalable Variational Gaussian Process Classification`,
        James Hensman, Alexander G. de G. Matthews, Zoubin Ghahramani

        [2] `Deep Gaussian Processes`,
        Andreas C. Damianou, Neil D. Lawrence

        :param torch.Tensor X: A input data for training. Its first dimension is the
            number of data points.
        :param torch.Tensor y: An output data for training. Its last dimension is the
            number of data points.
        """
        if y_1 is not None and X.shape[0] != y_1.shape[-1]:
            raise ValueError("Expected the number of input data points equal to the "
                             "number of output data points, but got {} and {}."
                             .format(X.shape[0], y_1.shape[-1]))
        if y_2 is not None and X.shape[0] != y_2.shape[-1]:
            raise ValueError("Expected the number of input data points equal to the "
                             "number of output data points, but got {} and {}."
                             .format(X.shape[0], y_2.shape[-1]))
        self.X = X
        self.y_1 = y_1
        self.y_2 = y_2

    def _check_Xnew_shape(self, Xnew):
        """
        Checks the correction of the shape of new data.

        :param torch.Tensor Xnew: A input data for testing. Note that
            ``Xnew.shape[1:]`` must be the same as ``self.X.shape[1:]``.
        """
        if Xnew.dim() != self.X.dim():
            raise ValueError("Train data and test data should have the same "
                             "number of dimensions, but got {} and {}."
                             .format(self.X.dim(), Xnew.dim()))
        if self.X.shape[1:] != Xnew.shape[1:]:
            raise ValueError("Train data and test data should have the same "
                             "shape of features, but got {} and {}."
                             .format(self.X.shape[1:], Xnew.shape[1:]))


def main():
    latent_dims = 2
    N = 500
    p = 50
    q = 50
    noise = 0.001
    n = int(N / 2)

    Z = np.random.rand(N, p)
    Z -= Z.mean(axis=0)
    Y = np.random.rand(N, q)
    Y -= Y.mean(axis=0)

    Z, Y, _, _ = generate_candola(N, latent_dims, p, q, noise, noise)

    Z_test, Y_test = Z[:n], Y[:n]
    Z, Y = Z[n:], Y[n:]

    lincca = linear.Wrapper().fit(Z, Y)

    pred_corr_lin = lincca.predict_corr(Z_test, Y_test)

    gpwrapped = Wrapper(outdim_size=latent_dims, sparse_x=True, sparse_y=True, jitter=1e-3)

    gpwrapped.fit(Z, Y)

    pred_corr_gp = gpwrapped.predict_corr(Z_test, Y_test)

    A = np.random.rand(N, p)
    A -= A.mean(axis=0)
    B = np.random.rand(N, q)
    B -= B.mean(axis=0)

    pred_corr_gp_rand = gpwrapped.predict_corr(A, B)

    print('test')


if __name__ == "__main__":
    main()
