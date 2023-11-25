"""
The :mod:`cca_zoo.datasets` module includes utilities to load datasets.
"""
from .simulated import JointData, LatentVariableData
from .toy import (
    load_breast_data,
    load_split_cifar10_data,
    load_mfeat_data,
    load_split_mnist_data,
)

__all__ = [
    "JointData",
    "LatentVariableData",
    "load_breast_data",
    "load_split_cifar10_data",
    "load_mfeat_data",
    "load_split_mnist_data",
]
