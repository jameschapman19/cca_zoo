import numbers
import logging
from jax._src import prng
from jax._src.random import PRNGKey
import numpy as np
import os
import gzip
import pandas as pd
from os.path import join
import jax.numpy as jnp

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


def get_num_batches(X, Y=None, batch_size=None):
    num = X.shape[0]
    if batch_size is None:
        batch_size = num
    num_complete_batches, leftover = divmod(num, batch_size)
    num_batches = num_complete_batches + bool(leftover)
    return num_batches


def data_stream(X, Y=None, batch_size=0, random_state=0):
    num = X.shape[0]
    if batch_size == 0:
        batch_size = num
    num_complete_batches, leftover = divmod(num, batch_size)
    num_batches = num_complete_batches + bool(leftover)
    rng = np.random.RandomState(random_state)
    while True:
        perm = rng.permutation(num)
        for i in range(num_batches):
            batch_idx = perm[i * batch_size : (i + 1) * batch_size]
            if Y is None:
                yield jnp.array(X[batch_idx])
            else:
                yield np.array(X[batch_idx]), np.array(Y[batch_idx])


def data_stream_UKBB(batch_ids, path, batch_size=0):
    num = len(batch_ids)
    if batch_size == 0:
        batch_size = num
    num_complete_batches, leftover = divmod(num, batch_size)
    num_batches = num_complete_batches + bool(leftover)
    rng = np.random.RandomState(0)
    while True:
        perm = rng.permutation(batch_ids)
        for i in range(num_batches):
            batch_idx = perm[i]
            # load batch - batches are in groups of 500 subjects
            # X is brain data
            X = (
                pd.read_csv(join(path, f"pack_{batch_idx}_img_sd.tab"), delimiter=" ")
                .to_numpy()
                .T
            )
            f = gzip.GzipFile(join(path, f"pack_{batch_idx}_norm.tab.gz"), "r")
            # Y is genetics data
            Y = pd.read_csv(f, delimiter=" ").to_numpy().T
            yield X, Y


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
