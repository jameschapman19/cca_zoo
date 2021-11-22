import scipy
from jax._src.numpy.util import _wraps
from functools import partial
from jax import jit
import jax.numpy as jnp
from jax._src.lax.linalg import schur

@partial(jit, static_argnames=('disp'))
def _logm(A, disp=False):
    keep_it_real = jnp.isrealobj(A)
    try:
        #if its upper triangular
        if jnp.array_equal(A, jnp.triu(A)):
            A = _logm_force_nonsingular_triangular_matrix(A)
            if jnp.min(jnp.diag(A)) < 0:
                A = A.astype(complex)
                return _logm_triu(A)
        #if its not upper triangular but it is real
        else:
            if keep_it_real:
                T, Z = schur(A)
            T = _logm_force_nonsingular_triangular_matrix(T, inplace=True)
            U = _logm_triu(T)
            ZH = jnp.conjugate(Z).T
            return Z.dot(U).dot(ZH)
    except:
        X = jnp.empty_like(A)
        X.fill(jnp.nan)
        return X

@_wraps(scipy.linalg.logm)
def logm(a, disp=False):
  return _logm(a,disp=disp)

@partial(jit, static_argnames=('inplace'))
def _logm_force_nonsingular_triangular_matrix(T, inplace=False):
    # The input matrix should be upper triangular.
    # The eps is ad hoc and is not meant to be machine precision.
    tri_eps = 1e-20
    abs_diag = jnp.absolute(jnp.diag(T))
    if jnp.any(abs_diag == 0):
        if not inplace:
            T = T.copy()
        n = T.shape[0]
        for i in range(n):
            if not T[i, i]:
                T[i, i] = tri_eps
    return T

@jit
def _logm_triu(T):
    """
    Compute matrix logarithm of an upper triangular matrix.
    The matrix logarithm is the inverse of
    expm: expm(logm(`T`)) == `T`
    Parameters
    ----------
    T : (N, N) array_like
        Upper triangular matrix whose logarithm to evaluate
    Returns
    -------
    logm : (N, N) ndarray
        Matrix logarithm of `T`
    References
    ----------
    .. [1] Awad H. Al-Mohy and Nicholas J. Higham (2012)
           "Improved Inverse Scaling and Squaring Algorithms
           for the Matrix Logarithm."
           SIAM Journal on Scientific Computing, 34 (4). C152-C169.
           ISSN 1095-7197
    .. [2] Nicholas J. Higham (2008)
           "Functions of Matrices: Theory and Computation"
           ISBN 978-0-898716-46-7
    .. [3] Nicholas J. Higham and Lijing lin (2011)
           "A Schur-Pade Algorithm for Fractional Powers of a Matrix."
           SIAM Journal on Matrix Analysis and Applications,
           32 (3). pp. 1056-1078. ISSN 0895-4798
    """
    T = jnp.asarray(T)
    if len(T.shape) != 2 or T.shape[0] != T.shape[1]:
        raise ValueError('expected an upper triangular square matrix')
    n, n = T.shape

    # Construct T0 with the appropriate type,
    # depending on the dtype and the spectrum of T.
    T_diag = jnp.diag(T)
    keep_it_real = jnp.isrealobj(T) and jnp.min(T_diag) >= 0
    if keep_it_real:
        T0 = T
    else:
        T0 = T.astype(complex)

    # Define bounds given in Table (2.1).
    theta = (None,
            1.59e-5, 2.31e-3, 1.94e-2, 6.21e-2,
            1.28e-1, 2.06e-1, 2.88e-1, 3.67e-1,
            4.39e-1, 5.03e-1, 5.60e-1, 6.09e-1,
            6.52e-1, 6.89e-1, 7.21e-1, 7.49e-1)

    R, s, m = _inverse_squaring_helper(T0, theta)

    # Evaluate U = 2**s r_m(T - I) using the partial fraction expansion (1.1).
    # This requires the nodes and weights
    # corresponding to degree-m Gauss-Legendre quadrature.
    # These quadrature arrays need to be transformed from the [-1, 1] interval
    # to the [0, 1] interval.
    nodes, weights = scipy.special.p_roots(m)
    nodes = nodes.real
    if nodes.shape != (m,) or weights.shape != (m,):
        raise Exception('internal error')
    nodes = 0.5 + 0.5 * nodes
    weights = 0.5 * weights
    ident = jnp.identity(n)
    U = jnp.zeros_like(R)
    for alpha, beta in zip(weights, nodes):
        U += solve_triangular(ident + beta*R, alpha*R)
    U *= jnp.exp2(s)

    # Skip this step if the principal branch
    # does not exist at T0; this happens when a diagonal entry of T0
    # is negative with imaginary part 0.
    has_principal_branch = all(x.real > 0 or x.imag != 0 for x in np.diag(T0))
    if has_principal_branch:

        # Recompute diagonal entries of U.
        U[jnp.diag_indices(n)] = jnp.log(jnp.diag(T0))

        # Recompute superdiagonal entries of U.
        # This indexing of this code should be renovated
        # when newer np.diagonal() becomes available.
        for i in range(n-1):
            l1 = T0[i, i]
            l2 = T0[i+1, i+1]
            t12 = T0[i, i+1]
            U[i, i+1] = _logm_superdiag_entry(l1, l2, t12)

    # Return the logm of the upper triangular matrix.
    if not jnp.array_equal(U, np.triu(U)):
        raise Exception('internal inconsistency')
    return U