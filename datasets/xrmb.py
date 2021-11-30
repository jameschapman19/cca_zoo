from scipy.io import loadmat

from ccagame.utils import data_stream
import jax.numpy as jnp

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

def xrmb_iterator(batch_size, n_components):
    X, Y, X_te, Y_te = xrmb()
    correct_U, _, correct_V = jnp.linalg.svd(X.T @ Y)
    correct_U = correct_U[:, :n_components]
    correct_V = correct_V[:n_components, :].T
    return data_stream(X, Y=Y, batch_size=batch_size), (X_te,Y_te), (correct_U, correct_V),(X.shape[1], Y.shape[1])
