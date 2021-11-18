import numbers

import numpy as np
from jax._src import prng
from jax._src.random import PRNGKey
from scipy.io import loadmat


def get_num_batches(X, Y=None, batch_size=None):
    num = X.shape[0]
    if batch_size is None:
        batch_size = num
    num_complete_batches, leftover = divmod(num, batch_size)
    num_batches = num_complete_batches + bool(leftover)
    return num_batches


def data_stream(views, batch_size=None):
    num = views[0].shape[0]
    if batch_size is None:
        batch_size = num
    num_complete_batches, leftover = divmod(num, batch_size)
    num_batches = num_complete_batches + bool(leftover)
    rng = np.random.RandomState(0)
    while True:
        perm = rng.permutation(num)
        for i in range(num_batches):
            batch_idx = perm[i * batch_size : (i + 1) * batch_size]
            yield [view[batch_idx] for view in views]


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
