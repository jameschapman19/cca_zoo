import numpy as np
from scipy.io import loadmat

from ._utils import demean, scale


def mediamill():
    """
    Download, parse and process MediaMill data
    Examples
    --------
    from ccagame import datasets

    train_view_1, train_view_2, test_view_1, test_view_2 = datasets.MediaMill()

    Returns
    -------
    train_view_1, train_view_2, test_view_1, test_view_2
    """
    try:
        project_dir = (
            "/home/jchapman/projects/blockeigengame/data/MediaMill/"
        )
        data = loadmat(project_dir + "mmill.mat")["data"][0][0]
    except:
        project_dir = "/home/chapm/blockeigengame/data/MediaMill/"
        data = loadmat(project_dir + "mmill.mat")["data"][0][0]
    view_1 = data["view1"][0][0]
    view_2 = data["view2"][0][0]
    return (
        view_1["training"].T,
        view_2["training"].T,
        view_1["tuning"].T,
        view_2["tuning"].T,
        view_1["testing"].T,
        view_2["testing"].T,
    )


def mediamill_dataset(model="cca"):
    X, Y, X_val, Y_val, X_te, Y_te = mediamill()
    X, X_te, Y, Y_te = scale(*demean(X, X_te, Y, Y_te))
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    X_te = X_te.astype(np.float32)
    Y_te = Y_te.astype(np.float32)
    return X, Y, X_te, Y_te


def mediamill_true(cca=False):
    if cca:
        try:
            U = np.load(
                "/home/jchapman/projects/blockeigengame/data/MediaMill/CCAU.npy"
            )
            V = np.load(
                "/home/jchapman/projects/blockeigengame/data/MediaMill/CCAV.npy"
            )
        except:
            U = np.load(
                "/home/chapm/blockeigengame/data/MediaMill/CCAU.npy"
            )
            V = np.load(
                "/home/chapm/blockeigengame/data/MediaMill/CCAV.npy"
            )
    else:
        try:
            U = np.load(
                "/home/jchapman/projects/blockeigengame/data/MediaMill/U.npy"
            )
            V = np.load(
                "/home/jchapman/projects/blockeigengame/data/MediaMill/V.npy"
            )
        except:
            U = np.load(
                "/home/chapm/blockeigengame/data/MediaMill/U.npy"
            )
            V = np.load(
                "/home/chapm/blockeigengame/data/MediaMill/V.npy"
            )
    return U, V
