import numpy as np
from scipy.linalg import toeplitz


def gaussian(x, mu, sig, dn):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) * dn / (np.sqrt(2 * np.pi) * sig)


def generate_mai(m, k, N, M, sparse_variables_1=None, sparse_variables_2=None, structure='identity', sigma=0.1):
    mean = np.zeros(N + M)
    cov = np.zeros((N + M, N + M))
    p = np.arange(0, k)
    p = 0.9 ** p
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
        c=np.arange(0, M)
        c = sigma ** c
        cov_2=toeplitz(c,c)

    cov[:N, :N] = cov_1
    cov[N:, N:] = cov_2

    # Sparse Bits
    up = np.random.rand(N, k)
    if sparse_variables_1 is not None:
        for _ in range(k):
            first = np.random.randint(N - 10)
            up[:first, _] = 0
            up[(first + 10):, _] = 0
            up /= np.sqrt((up.T @ cov_1 @ up)[_, _])

    vp = np.random.rand(M, k)
    if sparse_variables_2 is not None:
        for _ in range(k):
            first = np.random.randint(N - 10)
            vp[:first, _] = 0
            vp[(first + 10):, _] = 0
            vp /= np.sqrt((vp.T @ cov_2 @ vp)[_, _])
            # vp[np.random.choice(vp.shape[0], vp.shape[0]-sparse_variables_1, replace=False),_]=0

    sparse_vec = np.zeros((N, M))
    for _ in range(k):
        sparse_vec += 0.9 * p[_] * np.outer(up[:, _], vp[:, _])
    # Cross Bit
    cross = cov[:N, :N] @ sparse_vec @ cov[N:, N:]

    cov[N:, :N] = cross.T
    cov[:N, N:] = cross

    X = np.random.multivariate_normal(mean, cov, m)
    Y = X[:, N:]
    X = X[:, :N]
    return X, Y, up, vp


def generate_witten(m, k, N, M, sigma, tau, sparse_variables_1=2, sparse_variables_2=2):
    z = np.random.rand(m, k)

    up = np.random.rand(k, N)
    vp = np.random.rand(k, M)

    up[:, sparse_variables_1:] = 0
    vp[:, sparse_variables_2:] = 0

    X = z @ up + sigma * np.random.normal(0, 1, (m, N))
    Y = z @ vp + tau * np.random.normal(0, 1, (m, M))

    return X, Y, up.T, vp.T


def generate_candola(m, k, N, M, sigma, tau, sparse_variables_1=None, sparse_variables_2=None):
    # m data points
    # k dimensions
    # N unitary matrix size U
    # M unitary matrix size V
    X = np.random.rand(N, N)
    # in QR decomposition Q is orthogonal, R is upper triangular
    q, r = np.linalg.qr(X)
    # turns r into random 1s and -1s
    r = np.diag(np.diag(r) / np.abs(np.diag(r)))
    # returns a pxp matrix
    u = q @ r

    # check np.linalg.norm(u.T@u - np.linalg.eye(N))
    X = np.random.rand(M, M)
    q, r = np.linalg.qr(X)
    r = np.diag(np.diag(r) / np.abs(np.diag(r)))
    # returns a qxq matrix
    v = q @ r

    # returns mxk the latent space
    Z = np.random.rand(m, k)

    # extract first k columns from the
    up = u[:, : k]
    vp = v[:, : k]
    lam = np.zeros(N)
    for i in range(N):
        lam[i] = (((3 * N + 2) - 2 * i) / (2 * N))
    mu = np.zeros(M)
    for i in range(M):
        mu[i] = np.sqrt(((3 * M + 2) - 2 * i) / (2 * M))

    dL = np.diag(lam)
    dM = np.diag(mu)
    # NxN, Nxk (orthogonal columns), kxm
    X = (dL @ up @ Z.T) + (sigma * np.random.rand(N, m))
    # MxM, Mxk (orthogonal columns), kxm
    Y = (dM @ vp @ Z.T) + (tau * np.random.rand(M, m))
    return X.T, Y.T, up.T, vp.T
