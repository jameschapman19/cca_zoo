import os

import numpy as np
from sklearn.utils import Bunch

from cca_zoo._utils.check_values import check_rdata_support

DATA_MODULE = "cca_zoo.datasets.data"


def load_breast_data():
    # Describe File
    fdescr = ""
    check_rdata_support("load_breast_data")
    import rdata

    url = "https://tibshirani.su.domains/PMA/breastdata.rda"
    data_file_name = "breastdata.rda"

    # Download the data file
    tmpdir = os.path.join(os.getcwd(), "tmpdir")
    os.makedirs(tmpdir, exist_ok=True)
    filepath = os.path.join(tmpdir, data_file_name)

    if not os.path.exists(filepath):
        import urllib.request

        urllib.request.urlretrieve(url, filepath)

    parsed = rdata.parser.parse_file(filepath)
    converted = rdata.conversion.convert(parsed)["breastdata"]
    return Bunch(
        views=[converted["dna"], converted["rna"]],
        view_names=["dna", "rna"],
        chrom=converted["chrom"],
        nuc=converted["nuc"],
        gene=converted["gene"],
        genenames=converted["genenames"],
        genechr=converted["genechr"],
        genedesc=converted["genedesc"],
        genepos=converted["genepos"],
        DESCR=fdescr,
        filename=data_file_name,
        data_module=DATA_MODULE,
    )


def load_split_cifar10_data(data_home=None, cache=True):
    from sklearn.datasets import fetch_openml

    # Download CIFAR-10
    cifar_data = fetch_openml(name="CIFAR_10", data_home=data_home, cache=cache)

    # Split into left and right halves
    X = cifar_data.data.values

    # X is a 60000 x 3072 matrix. First 1024 columns are red, next 1024 are green, last 1024 are blue. The image is
    # stored in row-major order, so that the first 32 entries of the array are the red channel values of the first
    # row of the image. We reshape it to 60000 x 32 x 32 x 3 to get the RGB images.
    X_R = X[:, :1024].reshape((60000, 32, 32))
    X_G = X[:, 1024:2048].reshape((60000, 32, 32))
    X_B = X[:, 2048:].reshape((60000, 32, 32))
    X = np.stack((X_R, X_G, X_B), axis=3)
    X1 = X[:, :, :16, :]
    X2 = X[:, :, 16:, :]
    X1_R = X1[:, :, :, 0].reshape((60000, -1))
    X1_G = X1[:, :, :, 1].reshape((60000, -1))
    X1_B = X1[:, :, :, 2].reshape((60000, -1))
    X2_R = X2[:, :, :, 0].reshape((60000, -1))
    X2_G = X2[:, :, :, 1].reshape((60000, -1))
    X2_B = X2[:, :, :, 2].reshape((60000, -1))
    X1 = np.hstack((X1_R, X1_G, X1_B))
    X2 = np.hstack((X2_R, X2_G, X2_B))
    cifar_data.views = [X1, X2]
    return cifar_data


def load_mfeat_data(features=None):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mfeat/mfeat.tar"
    data_file_name = "mfeat.tar"
    # Download the data file
    tmpdir = os.path.join(os.getcwd(), "tmpdir")
    os.makedirs(tmpdir, exist_ok=True)
    filepath = os.path.join(tmpdir, data_file_name)
    from torchvision.datasets.utils import download_and_extract_archive

    if not os.path.exists(filepath):
        download_and_extract_archive(url, download_root=tmpdir, filename=data_file_name)
    if features is None:
        features = ["fac", "fou", "kar", "mor", "pix", "zer"]
    views = [
        np.genfromtxt(os.path.join(tmpdir, f"mfeat/mfeat-{feature}"))
        for feature in features
    ]
    # first 200 patterns are of class `0', followed by sets of 200 patterns
    # for each of the classes `1' - `9'.
    targets = np.array(
        [0] * 200
        + [1] * 200
        + [2] * 200
        + [3] * 200
        + [4] * 200
        + [5] * 200
        + [6] * 200
        + [7] * 200
        + [8] * 200
        + [9] * 200
    )
    return Bunch(
        views=views,
        target=targets,
        DESCR="MFeat Dataset",
        data_module=DATA_MODULE,
    )


def load_split_mnist_data():
    from sklearn.datasets import fetch_openml

    # Download MNIST
    mnist_data = fetch_openml(name="mnist_784")

    # Split into left and right halves
    X = mnist_data.data.values
    X1 = X[:, :392]
    X2 = X[:, 392:]
    mnist_data.views = [X1, X2]
    return mnist_data
