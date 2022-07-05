import gzip
import jax.numpy as jnp
import logging
import numbers
import numpy as np
import os
import pandas as pd
from functools import partial
from jax import jit
from jax._src import prng
from jax._src.random import PRNGKey
from os.path import join
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


@jit
def _get_AB(X_i, Y_i):
    p = X_i.shape[1]
    n = X_i.shape[0]
    C = jnp.hstack((X_i, Y_i)).T @ jnp.hstack((X_i, Y_i)) / n
    A = C.at[:p, :p].set(0)
    A = A.at[p:, p:].set(0)
    B = C.at[:p, p:].set(0)
    B = B.at[p:, :p].set(0)
    return A, B


def invsqrtm(C):
    return jnp.linalg.inv(sqrtm(C))
