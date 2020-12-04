import numpy as np
from scipy import linalg

def chol_sample(mean, chol):
    return mean + chol @ np.random.standard_normal(mean.size)

def gaussian(x, mu, sig, dn):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) * dn / (np.sqrt(2 * np.pi) * sig)

def generate_mai(m: int, k: int, N: int, M: int, sparse_variables_1: float = 0, sparse_variables_2: float = 0,
                 signal: float = 1,
                 structure: str = 'identity', sigma: float = 0.9, decay: float = 0.5, rand_eigs_1: bool = False,
                 rand_eigs_2: bool = False):
    """
    :param m: number of samples
    :param k: number of latent dimensions
    :param N: number of features in view 1
    :param M: number of features in view 2
    :param sparse_variables_1: fraction of active variables from view 1 associated with true signal
    :param sparse_variables_2: fraction of active variables from view 2 associated with true signal
    :param signal: correlation
    :param structure: within view covariance structure
    :param sigma:
    :param decay:
    :param rand_eigs_1:
    :param rand_eigs_2:
    :return: tuple of numpy arrays: view_1, view_2, true weights from view 1, true weights from view 2, overall covariance structure
    """
    mean = np.zeros(N + M)
    cov = np.zeros((N + M, N + M))
    p = np.arange(0, k)
    p = decay ** p
    # Covariance Bit
    if structure == 'identity':
        cov_1 = np.eye(N)
        cov_2 = np.eye(M)
    elif structure == 'gaussian':
        x = np.linspace(-1, 1, N)
        x_tile = np.tile(x, (N, 1))
        mu_tile = np.transpose(x_tile)
        dn = 2 / (N - 1)
        cov_1 = gaussian(x_tile, mu_tile, sigma, dn)
        cov_1 /= cov_1.max()
        x = np.linspace(-1, 1, M)
        x_tile = np.tile(x, (M, 1))
        mu_tile = np.transpose(x_tile)
        dn = 2 / (M - 1)
        cov_2 = gaussian(x_tile, mu_tile, sigma, dn)
        cov_2 /= cov_2.max()
    elif structure == 'toeplitz':
        c = np.arange(0, N)
        c = sigma ** c
        cov_1 = linalg.toeplitz(c, c)
        c = np.arange(0, M)
        c = sigma ** c
        cov_2 = linalg.toeplitz(c, c)
    elif structure == 'random':
        if rand_eigs_1:
            cov_1 = np.random.rand(N, N)
            U, S, V = np.linalg.svd(cov_1.T @ cov_1)
            cov_1 = U @ (np.diag(np.random.rand(N))) @ V
        else:
            cov_1 = np.random.rand(N, N)
            cov_1 = cov_1.T @ cov_1
            # cov_1 = make_sparse_spd_matrix(N, alpha=0.7)
        if rand_eigs_2:
            cov_2 = np.random.rand(M, M)
            U, S, V = np.linalg.svd(cov_2.T @ cov_2)
            cov_2 = U @ (np.diag(np.random.rand(M))) @ V
        else:
            cov_2 = np.random.rand(M, M)
            cov_2 = cov_2.T @ cov_2
            # cov_2 = make_sparse_spd_matrix(M, alpha=0.7)
    cov[:N, :N] = cov_1
    cov[N:, N:] = cov_2
    del cov_1
    del cov_2

    up = np.random.rand(N, k) - 0.5
    for _ in range(k):
        if sparse_variables_1 > 0:
            if sparse_variables_1 < 1:
                sparse_variables_1 = np.ceil(sparse_variables_1 * N).astype('int')
            first = np.random.randint(N - sparse_variables_1)
            up[:first, _] = 0
            up[(first + sparse_variables_1):, _] = 0
        up[:, _] /= np.sqrt((up[:, _].T @ cov[:N, :N] @ up[:, _]))

    vp = np.random.rand(M, k) - 0.5
    for _ in range(k):
        if sparse_variables_2 > 0:
            if sparse_variables_2 < 1:
                sparse_variables_2 = np.ceil(sparse_variables_2 * M).astype('int')
            first = np.random.randint(M - sparse_variables_2)
            vp[:first, _] = 0
            vp[(first + sparse_variables_2):, _] = 0
        vp[:, _] /= np.sqrt((vp[:, _].T @ cov[N:, N:] @ vp[:, _]))

    cross = np.zeros((N, M))
    for _ in range(k):
        cross += signal * p[_] * np.outer(up[:, _], vp[:, _])
    # Cross Bit
    cross = cov[:N, :N] @ cross @ cov[N:, N:]

    cov[N:, :N] = cross.T
    cov[:N, N:] = cross
    del cross

    if cov.shape[0] < 2000:
        X = np.random.multivariate_normal(mean, cov, m)
    else:
        X = np.zeros((m, N + M))
        chol = np.linalg.cholesky(cov)
        for _ in range(m):
            X[_, :] = chol_sample(mean, chol)
    Y = X[:, N:]
    X = X[:, :N]

    return X, Y, up, vp, cov
