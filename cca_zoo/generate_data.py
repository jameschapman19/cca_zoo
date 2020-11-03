import numpy as np
from scipy.linalg import toeplitz


def gaussian(x, mu, sig, dn):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) * dn / (np.sqrt(2 * np.pi) * sig)


def generate_mai(m: int, k: int, N: int, M: int, sparse_variables_1: float = 0, sparse_variables_2: float = 0,
                 signal: float = 1,
                 structure: str = 'identity', sigma: float = 0.9, decay: float = 0.5):
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
        cov_1 = toeplitz(c, c)
        c = np.arange(0, M)
        c = sigma ** c
        cov_2 = toeplitz(c, c)
    elif structure == 'random':
        cov_1 = np.random.rand(N, N)
        # cov_1=cov_1@cov_1.T
        U, S, V = np.linalg.svd(cov_1.T @ cov_1)
        cov_1 = U @ (1.0 + np.diag(np.random.rand(N))) @ V
        cov_2 = np.random.rand(M, M)
        # cov_2=cov_2@cov_2.T
        U, S, V = np.linalg.svd(cov_2.T @ cov_2)
        cov_2 = U @ (1.0 + np.diag(np.random.rand(M))) @ V

    cov[:N, :N] = cov_1
    cov[N:, N:] = cov_2
    del cov_1
    del cov_2

    """
    # Sparse Bits
    if sparse_variables_1 > 0:
        sparse_cov_1 = csr_matrix(cov_1)
        cov_1 = sparse_cov_1.copy()
    else:
        sparse_cov_1 = cov_1.copy()
    """
    up = np.random.rand(N, k) - 0.5
    for _ in range(k):
        if sparse_variables_1 > 0:
            if sparse_variables_1 < 1:
                sparse_variables_1 = np.ceil(sparse_variables_1 * N).astype('int')
            first = np.random.randint(N - sparse_variables_1)
            up[:first, _] = 0
            up[(first + sparse_variables_1):, _] = 0
        up[:, _] /= np.sqrt((up[:, _].T @ cov[:N, :N] @ up[:, _]))
        """
        if _ < (k - 1) and sparse_variables_1 == 0:
            proj = csr_matrix(up[:, _]).T @ csr_matrix(up[:, _])
            cov_1 = (identity(up[:, _].shape[0]) - proj) @ cov_1 @ (identity(up[:, _].shape[0]) - proj)
        """

    """
    # Elimination step:
    for _ in range(k):
        mat_1 = up.T @ sparse_cov_1 @ up
        up[:, (_ + 1):] -= np.outer(up[:, _], mat_1[_, (_ + 1):])
    """

    """
    if sparse_variables_2 > 0:
        sparse_cov_2 = csr_matrix(cov_2)
        cov_2 = sparse_cov_2.copy()
    else:
        sparse_cov_2 = cov_2.copy()
    """
    vp = np.random.rand(M, k) - 0.5
    for _ in range(k):
        if sparse_variables_2 > 0:
            if sparse_variables_2 < 1:
                sparse_variables_2 = np.ceil(sparse_variables_2 * M).astype('int')
            first = np.random.randint(M - sparse_variables_2)
            vp[:first, _] = 0
            vp[(first + sparse_variables_2):, _] = 0
        vp[:, _] /= np.sqrt((vp[:, _].T @ cov[N:, N:] @ vp[:, _]))
        """
        if _ < (k - 1) and sparse_variables_2 == 0:
            proj = csr_matrix(vp[:, _]).T @ csr_matrix(vp[:, _])
            cov_2 = (identity(vp[:, _].shape[0]) - proj) @ cov_2 @ (identity(vp[:, _].shape[0]) - proj)
        """
    """
    for _ in range(k):
        mat_2 = vp.T @ sparse_cov_2 @ vp
        vp[:, (_ + 1):] -= np.outer(vp[:, _], mat_2[_, (_ + 1):])
    """

    cross = np.zeros((N, M))
    for _ in range(k):
        cross += signal * p[_] * np.outer(up[:, _], vp[:, _])
    # Cross Bit
    cross = cov[:N, :N] @ cross @ cov[N:, N:]

    cov[N:, :N] = cross.T
    cov[:N, N:] = cross
    del cross

    X = np.random.multivariate_normal(mean, cov, m)
    Y = X[:, N:]
    X = X[:, :N]

    return X, Y, up, vp
