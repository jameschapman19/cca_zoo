"""Helped by https://github.com/bcdutton/AdversarialCanonicalCorrelationAnalysis (hopefully I will have my own
implementation of their work soon) Check out their paper at https://arxiv.org/abs/2005.10349 """

import numpy as np
import torch
import torch.utils.data
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms


class SplitMNISTDataset(Dataset):
    """
    Class to generate paired noisy mnist data
    """

    def __init__(
        self,
        root: str,
        mnist_type: str = "MNIST",
        train: bool = True,
        flatten: bool = True,
        download=False,
    ):
        """
        :param root: Root directory of dataset
        :param mnist_type: "MNIST", "FashionMNIST" or "KMNIST"
        :param train: whether this is train or test
        :param flatten: whether to flatten the data into array or use 2d images
        :param download: If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        """

        self.dataset = load_mnist(mnist_type, train, root, download)
        self.flatten = flatten

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x_a, label = self.dataset[idx]
        x_b = x_a[:, :, 14:]
        x_a = x_a[:, :, :14]
        if self.flatten:
            x_a = torch.flatten(x_a)
            x_b = torch.flatten(x_b)
        return {"views": (x_a, x_b), "label": label, "index": idx}


class NoisyMNISTDataset(Dataset):
    """
    Class to generate paired noisy mnist data
    """

    def __init__(
        self,
        root: str,
        mnist_type: str = "MNIST",
        train: bool = True,
        flatten: bool = True,
        download=False,
    ):
        """
        :param root: Root directory of dataset
        :param mnist_type: "MNIST", "FashionMNIST" or "KMNIST"
        :param train: whether this is train or test
        :param download: If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        :param flatten: whether to flatten the data into array or use 2d images
        """
        self.dataset = load_mnist(mnist_type, train, root, download)
        self.a_transform = torchvision.transforms.RandomRotation((-45, 45))
        self.b_transform = transforms.Compose(
            [
                transforms.Lambda(_add_mnist_noise),
                transforms.Lambda(self.__threshold_func__),
            ]
        )
        self.targets = self.dataset.targets
        self.filtered_classes = []
        self.filtered_nums = []
        for i in range(10):
            self.filtered_nums.append(np.where(self.targets == i)[0])
        self.flatten = flatten

    def __threshold_func__(self, x):
        x[x > 1] = 1
        return x

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x_a, label = self.dataset[index]
        x_a = self.a_transform(x_a)
        # get random index of image with same class
        random_index = np.random.choice(self.filtered_nums[label])
        x_b = self.b_transform(self.dataset[random_index][0])
        if self.flatten:
            x_a = torch.flatten(x_a)
            x_b = torch.flatten(x_b)
        return {"views": (x_a, x_b), "label": label, "index": index}


class TangledMNISTDataset(Dataset):
    """
    Class to generate paired tangled MNIST dataset
    """

    def __init__(
        self,
        root: str,
        mnist_type: str = "MNIST",
        train: bool = True,
        flatten: bool = True,
        download=False,
    ):
        """
        :param root: Root directory of dataset
        :param mnist_type: "MNIST", "FashionMNIST" or "KMNIST"
        :param train: whether this is train or test
        :param flatten: whether to flatten the data into array or use 2d images
        :param download: If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        """
        self.dataset = load_mnist(mnist_type, train, root, download)
        self.transform = torchvision.transforms.RandomRotation((-45, 45))
        self.targets = self.dataset.targets
        self.filtered_classes = []
        self.filtered_nums = []
        for i in range(10):
            self.filtered_nums.append(np.where(self.targets == i)[0])
        self.flatten = flatten

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x_a, label = self.dataset[idx]
        x_a = self.transform(x_a)
        # get random index of image with same class
        random_index = np.random.choice(self.filtered_nums[label])
        x_b = self.transform(self.dataset[random_index][0])
        if self.flatten:
            x_a = torch.flatten(x_a)
            x_b = torch.flatten(x_b)
        return {"views": (x_a, x_b), "label": label, "index": idx}


def _add_mnist_noise(x):
    x = x + torch.rand(28, 28) / 10
    return x


def load_mnist(mnist_type, train, root, download):
    if mnist_type == "MNIST":
        dataset = datasets.MNIST(
            root,
            train=train,
            download=download,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()]
            ),
        )

    elif mnist_type == "FashionMNIST":
        dataset = datasets.FashionMNIST(
            root,
            train=train,
            download=download,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()]
            ),
        )
    elif mnist_type == "KMNIST":
        dataset = datasets.KMNIST(
            root,
            train=train,
            download=download,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                ]
            ),
        )
    else:
        raise ValueError
    return dataset
