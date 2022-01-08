from scipy.io import loadmat
import numpy as np
import os


def xrmb():
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
    try:
        project_dir = "/home/chapmajw/ccagame/XRMB/"
        view_1 = loadmat(project_dir + "XRMBf2KALDI_window7_single1.mat")
        view_2 = loadmat(project_dir + "XRMBf2KALDI_window7_single2.mat")
    except:
        project_dir = "/home/jchapman/projects/ccagame/XRMB"
        view_1 = loadmat(project_dir + "XRMBf2KALDI_window7_single1.mat")
        view_2 = loadmat(project_dir + "XRMBf2KALDI_window7_single2.mat")
    return view_1["X1"], view_2["X2"], view_1["XTe1"], view_2["XTe2"]


def xrmb_dataset():
    X, Y, X_te, Y_te = xrmb()
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    X_te = X_te.astype(np.float32)
    Y_te = Y_te.astype(np.float32)
    return X, Y, X_te, Y_te
