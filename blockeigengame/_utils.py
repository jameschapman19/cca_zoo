import logging
import numbers
import os
from functools import partial

import jax.numpy as jnp
from jax import jit
from jax._src import prng
from jax._src.random import PRNGKey
from scipy.linalg import sqrtm

log = logging.getLogger(__name__)


def _get_next_version():
    root_dir = os.getcwd()

    try:
        listdir_info = os.listdir(root_dir)
    except OSError:
        log.warning("Missing logger folder: %s", root_dir)
        return 0

    existing_versions = []
    for listing in listdir_info:
        if os.path.isdir(listing) and listing.startswith("version_"):
            dir_ver = listing.split("_")[1].replace("/", "")
            existing_versions.append(int(dir_ver))
    if len(existing_versions) == 0:
        return 0
    return max(existing_versions) + 1


def log_dir(version=None) -> str:
    """The directory for this run's tensorboard checkpoint.
    By default, it is named ``'version_${self.version}'`` but it can be overridden by passing a string value for the
    constructor's version parameter instead of ``None`` or an int.
    """
    if version is None:
        version = _get_next_version()
    # create a pseudo standard path ala test-tube
    version = version if isinstance(version, str) else f"version_{version}"
    log_dir = os.path.join(os.getcwd(), version)
    log_dir = os.path.expandvars(log_dir)
    log_dir = os.path.expanduser(log_dir)
    os.mkdir(log_dir)
    return log_dir


def check_random_state(seed):
    """Turn seed into a prng. instance
    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None:
        return PRNGKey(0)
    if isinstance(seed, numbers.Integral):
        return PRNGKey(seed)
    if isinstance(seed, prng.PRNGKeyArray):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )


@partial(jit, static_argnums=(1))
def _split_eigenvector(W, dim):
    return W[:, :dim], W[:, dim:]


def invsqrtm(C):
    return jnp.linalg.inv(sqrtm(C))


@jit
def incrsvd(x_projected, y_projected, x_leftover, y_leftover, U, V, S):
    n_components = S.shape[0]
    Q = jnp.vstack(
        (
            jnp.hstack(
                (
                    jnp.diag(S) + x_projected.T @ y_projected,
                    jnp.linalg.norm(y_leftover, axis=1).T * x_projected.T,
                )
            ),
            jnp.hstack(
                (
                    (jnp.linalg.norm(x_leftover, axis=1).T * y_projected.T).T,
                    jnp.atleast_2d(
                        jnp.linalg.norm(x_leftover, axis=1, keepdims=True)
                        @ jnp.linalg.norm(y_leftover, axis=1, keepdims=True).T
                    ),
                )
            ),
        )
    )
    U_, S, Vt_ = jnp.linalg.svd(Q)
    U = U_[:, :n_components].T @ jnp.vstack((U, x_leftover / jnp.linalg.norm(x_leftover, axis=1, keepdims=True)))
    V = Vt_.T[:, :n_components].T @ jnp.vstack((V, y_leftover / jnp.linalg.norm(y_leftover, axis=1, keepdims=True)))
    S = S[:n_components]
    return U, V, S


def _capping(S, k):
    S_tmp = S.copy()
    S_tmp = jnp.where(S_tmp > 1, 1, S_tmp)
    S_tmp = jnp.where(S_tmp < 0, 0, S_tmp)
    trace = S_tmp.sum()
    if trace < k:
        return S_tmp
    else:
        return _capping_loop(S, k)


@partial(jit, static_argnums=(1))
def _capping_loop(S, k):
    for i in range(S.shape[0]):
        for j in range(i, S.shape[0]):
            S_tmp = jnp.flip(S.copy())
            if i > 0:
                S_tmp = S_tmp.at[:i].set(0)
            if j < S.shape[0] - 2:
                S_tmp = S_tmp.at[j + 1:].set(1)
            shf = k - S_tmp.sum() / (j - i + 1)
            S_tmp = S_tmp.at[i:j].set(S_tmp[i:j] + shf)
            if S_tmp[i] >= 0 and (i == 0 or (jnp.flip(S)[(i - 1)] + shf) <= 0) and S_tmp[j] <= 1 and (
                    j == S.shape[0] - 1 or (jnp.flip(S)[j] + shf) >= 1):
                return jnp.flip(S_tmp)
