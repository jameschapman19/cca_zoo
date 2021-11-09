import numbers

import numpy as np
from jax._src import prng
from jax._src.random import PRNGKey
from scipy.io import loadmat


def get_xrmb(
    datadir="/mnt/c/Users/chapm/PycharmProjects/ccagame/data/XRMB/", mode="train"
):
    view_1 = loadmat(datadir + "XRMBf2KALDI_window7_single1.mat")
    view_2 = loadmat(datadir + "XRMBf2KALDI_window7_single2.mat")
    if mode == "train":
        view_1 = view_1["X1"]
        view_2 = view_2["X2"]
    elif mode == "val":
        view_1 = view_1["XV1"]
        view_2 = view_2["XV2"]
    elif mode == "test":
        view_1 = view_1["XTe1"]
        view_2 = view_2["XTe2"]
    return view_1, view_2


def get_num_batches(X, Y=None, batch_size=None):
    num = X.shape[0]
    if batch_size is None:
        batch_size = num
    num_complete_batches, leftover = divmod(num, batch_size)
    num_batches = num_complete_batches + bool(leftover)
    return num_batches


def data_stream(X, Y=None, batch_size=None):
    num = X.shape[0]
    if batch_size is None:
        batch_size = num
    num_complete_batches, leftover = divmod(num, batch_size)
    num_batches = num_complete_batches + bool(leftover)
    rng = np.random.RandomState(0)
    while True:
        perm = rng.permutation(num)
        for i in range(num_batches):
            batch_idx = perm[i * batch_size : (i + 1) * batch_size]
            if Y is None:
                yield X[batch_idx]
            else:
                yield X[batch_idx], Y[batch_idx]


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
