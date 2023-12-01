import os

import numpy as np
from sklearn.utils import Bunch

from cca_zoo._utils._checks import check_rdata_support

DATA_MODULE = "cca_zoo.datasets.data"


def load_breast_data():
    """
    Loads the breast data from a remote .rda file.

    This function fetches the 'breastdata.rda' dataset from the specified URL,
    parses the R data file, and returns the dataset as a Bunch object containing
    various attributes related to the breast data.

    The function checks for RData support and uses the `rdata` library for parsing.
    If the file does not exist locally, it is downloaded and stored in a temporary
    directory.

    Returns:
        Bunch: An object with the following attributes:
            - views (list): Contains two arrays representing 'dna' and 'rna'.
            - view_names (list): Names of the views, i.e., ["dna", "rna"].
            - chrom (array): Information related to chromosomal locations.
            - nuc (array): Nucleotide sequences.
            - gene (array): Gene sequences.
            - genenames (array): Names of the genes.
            - genechr (array): Chromosomal locations for each gene.
            - genedesc (array): Descriptions of the genes.
            - genepos (array): Positional information for each gene.
            - DESCR (str): Description of the dataset (currently empty).
            - filename (str): Name of the R data file, i.e., 'breastdata.rda'.
            - data_module: Reference to the data module (assumed to be a global constant).

    Notes:
        - Ensure the `rdata` library is installed and functional.
        - The data is fetched from 'https://tibshirani.su.domains/PMA/breastdata.rda'.
        - The temporary directory 'tmpdir' is created in the current working directory if it doesn't exist.

    Raises:
        SomeException: Description of under what condition an exception is raised.
        (You would fill in `SomeException` with the actual exception(s) that might be raised, if any).

    Example:
        data = load_breast_data()
        print(data.views)
        print(data.view_names)
        # ... and so on for other attributes ...

    """
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
    """
    Load and split the CIFAR-10 dataset into two halves based on color channels.

    Parameters:
    - data_home (str or None, optional): The directory where the CIFAR-10 dataset will be cached.
    If None, the default Scikit-learn cache directory will be used.
    - cache (bool, optional): Whether to cache the dataset for faster access.

    Returns:
    - cifar_data (Bunch object): A Scikit-learn Bunch object containing the CIFAR-10 dataset.
    This object has 'data' and 'target' attributes.

    The function fetches the CIFAR-10 dataset from Scikit-learn's dataset repository and splits it into two halves:
    - The first half, X1, contains images with the left 16x32 pixel region (red channel, green channel, and blue channel).
    - The second half, X2, contains images with the right 16x32 pixel region (red channel, green channel, and blue channel).

    Each channel of the images is further reshaped and concatenated to form the final feature matrices X1 and X2.

    The Bunch object cifar_data also stores these views as 'views' attribute.

    Note:
    - CIFAR-10 is a dataset of 60,000 32x32 color images in 10 different classes, with 6,000 images per class.
    - The images in CIFAR-10 are stored in row-major order, where each row contains the pixel values for a 32x32 image.

    Example usage:
    >>> cifar_data = load_split_cifar10_data()
    >>> X1, X2 = cifar_data.views
    >>> print(X1.shape)  # Shape of the first half of the dataset
    >>> print(X2.shape)  # Shape of the second half of the dataset
    """

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
    """
    Load the Multiple Features (MFeat) dataset.

    Parameters:
    - features (list of str or None, optional): List of feature types to load.
    Available feature types: ["fac", "fou", "kar", "mor", "pix", "zer"].
    If None, all available features will be loaded.

    Returns:
    - data (Bunch object): A Scikit-learn Bunch object containing the MFeat dataset.
    This object has 'views' (features), 'target' (class labels), 'DESCR' (description),
    and 'data_module' attributes.

    The function downloads and extracts the MFeat dataset from a remote URL if it's not already
    downloaded. It allows you to specify which types of features to load (e.g., "fac", "fou", etc.).

    The MFeat dataset consists of multiple sets of features, with each set corresponding to
    a different class. The first 200 patterns are of class '0', followed by sets of 200 patterns
    for each of the classes '1' to '9'.

    Example usage:
    >>> mfeat_data = load_mfeat_data(features=["fac", "fou"])
    >>> features = mfeat_data.views
    >>> labels = mfeat_data.target
    >>> print(features[0].shape)  # Shape of the loaded features
    >>> print(labels.shape)       # Shape of the loaded labels
    """
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
    """
    Load and split the MNIST dataset into two halves.

    Returns:
    - mnist_data (Bunch object): A Scikit-learn Bunch object containing the MNIST dataset.
      This object has 'views' attribute, where 'views[0]' corresponds to the left half of the images,
      and 'views[1]' corresponds to the right half of the images.

    The function fetches the MNIST dataset from Scikit-learn's dataset repository and splits each
    28x28 pixel image into two halves:
    - The first half (left), X1, contains the first 14 columns (left 14 pixels) of each image.
    - The second half (right), X2, contains the last 14 columns (right 14 pixels) of each image.

    The Bunch object mnist_data also stores these views as 'views' attribute.

    Example usage:
    >>> mnist_data = load_split_mnist_data()
    >>> left_half = mnist_data.views[0]
    >>> right_half = mnist_data.views[1]
    >>> print(left_half.shape)   # Shape of the left half of the dataset
    >>> print(right_half.shape)  # Shape of the right half of the dataset
    """
    from sklearn.datasets import fetch_openml

    # Download MNIST
    mnist_data = fetch_openml(name="mnist_784")

    # Split into left and right halves
    X = mnist_data.data.values
    X1 = X[:, :392]
    X2 = X[:, 392:]
    mnist_data.views = [X1, X2]
    return mnist_data
