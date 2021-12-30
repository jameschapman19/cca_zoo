from cca_zoo.models.rcca import rCCA
from scipy.io import loadmat

from ccagame.utils import data_stream
import jax.numpy as jnp
import numpy as np


def xrmb(datadir="/mnt/c/Users/chapm/PycharmProjects/barlowtwins/data/XRMB/"):
    """
    Download, parse and process xrmb data
    Examples
    --------
    from ccagame import datasets

    train_view_1, train_view_2, test_view_1, test_view_2 = datasets.xrmb()

    Returns
    -------
    train_view_1, train_view_2, test_view_1, test_view_2
    """
    view_1 = loadmat(datadir + "XRMBf2KALDI_window7_single1.mat")
    view_2 = loadmat(datadir + "XRMBf2KALDI_window7_single2.mat")

    return view_1["X1"], view_2["X2"], view_1["XTe1"], view_2["XTe2"]


def xrmb_dataset():
    X, Y, X_te, Y_te = xrmb()
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    X_te = X_te.astype(np.float32)
    Y_te = Y_te.astype(np.float32)
    return X, Y, X_te, Y_te
