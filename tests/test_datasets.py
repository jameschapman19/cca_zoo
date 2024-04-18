from cca_zoo.datasets import load_breast_data
from cca_zoo.datasets import load_mfeat_data
from cca_zoo.datasets import load_split_mnist_data


def test_load_breast_data():
    data = load_breast_data()
    assert data is not None
    assert data.views is not None
    assert data.view_names is not None
    assert data.chrom is not None
    assert data.nuc is not None
    assert data.gene is not None
    assert data.genenames is not None
    assert data.genechr is not None
    assert data.genedesc is not None
    assert data.genepos is not None
    assert data.DESCR is not None
    assert data.filename is not None
    assert data.data_module == "cca_zoo.datasets.data"


# def test_load_split_cifar10_data():
#     data = load_split_cifar10_data()
#     assert data is not None
#     assert data.views is not None
#     assert data.target is not None
#     assert data.feature_names is not None
#     assert data.target_names is not None
#     assert data.DESCR is not None


# def test_load_mfeat_data():
#     data = load_mfeat_data()
#     assert data is not None
#     assert data.views is not None
#     assert data.target is not None
#     assert data.DESCR is not None
#     assert data.data_module == "cca_zoo.datasets.data"


def test_load_split_mnist_data():
    data = load_split_mnist_data()
    assert data is not None
    assert data.views is not None
    assert data.target is not None
    assert data.DESCR is not None
