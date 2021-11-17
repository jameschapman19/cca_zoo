"""Helped by https://github.com/bcdutton/AdversarialCanonicalCorrelationAnalysis (hopefully I will have my own
implementation of their work soon) Check out their paper at https://arxiv.org/abs/2005.10349 """

import numpy as np
import torch
import torch.utils.data
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class Split_MNIST_Dataset(Dataset):
    """
    Class to generate paired noisy mnist data
    """

    def __init__(
            self, mnist_type: str = "MNIST", train: bool = True, flatten: bool = True
    ):
        """

        :param mnist_type: "MNIST", "FashionMNIST" or "KMNIST"
        :param train: whether this is train or test
        :param flatten: whether to flatten the data into array or use 2d images
        """
        if mnist_type == "MNIST":
            self.dataset = datasets.MNIST("../../data", train=train, download=True)
        elif mnist_type == "FashionMNIST":
            self.dataset = datasets.FashionMNIST(
                "../../data", train=train, download=True
            )
        elif mnist_type == "KMNIST":
            self.dataset = datasets.KMNIST("../../data", train=train, download=True)

        self.data = self.dataset.data
        self.targets = self.dataset.targets
        self.flatten = flatten

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx].flatten()
        x_a = x[:392] / 255
        x_b = x[392:] / 255
        label = self.targets[idx]
        return (x_a, x_b), label

    def to_numpy(self, indices=None):
        """
        Converts dataset to numpy array form

        :param indices: indices of the samples to extract into numpy arrays
        """
        if indices is None:
            indices = np.arange(self.__len__())
        view_1 = np.zeros((len(indices), 392))
        view_2 = np.zeros((len(indices), 392))
        labels = np.zeros(len(indices)).astype(int)
        for i, n in enumerate(indices):
            sample = self[n]
            view_1[i] = sample[0][0].numpy()
            view_2[i] = sample[0][1].numpy()
            labels[i] = sample[1].numpy().astype(int)
        return (view_1, view_2), labels


class Noisy_MNIST_Dataset(Dataset):
    """
    Class to generate paired noisy mnist data
    """

    def __init__(
            self, mnist_type: str = "MNIST", train: bool = True, flatten: bool = True
    ):
        """

        :param mnist_type: "MNIST", "FashionMNIST" or "KMNIST"
        :param train: whether this is train or test
        :param flatten: whether to flatten the data into array or use 2d images
        """
        if mnist_type == "MNIST":
            self.dataset = datasets.MNIST("../../data", train=train, download=True,
                                          transform=torchvision.transforms.Compose([
                                              torchvision.transforms.ToTensor()]))
        elif mnist_type == "FashionMNIST":
            self.dataset = datasets.FashionMNIST(
                "../../data", train=train, download=True, transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor()]))
        elif mnist_type == "KMNIST":
            self.dataset = datasets.KMNIST("../../data", train=train, download=True,
                                           transform=torchvision.transforms.Compose([
                                               torchvision.transforms.ToTensor(),
                                               transforms.Normalize((0.1307,), (0.3081,))]))

        self.base_transform = transforms.ToTensor()
        self.a_transform = transforms.Compose(
            [
                torchvision.transforms.RandomRotation((-45, 45))
            ]
        )
        self.a_transform = transforms.Compose(
            [
                torchvision.transforms.RandomRotation((-45, 45))
            ]
        )
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

    def __getitem__(self, idx):
        x_a, label = self.dataset[idx]
        x_a = self.a_transform(x_a)
        # get random index of image with same class
        random_index = np.random.choice(self.filtered_nums[label])
        x_b = self.b_transform(self.dataset[random_index][0])
        if self.flatten:
            x_a = torch.flatten(x_a)
            x_b = torch.flatten(x_b)
        return (x_b, x_a), label


class Tangled_MNIST_Dataset(Dataset):
    """
    Class to generate paired tangled MNIST dataset
    """

    def __init__(self, mnist_type="MNIST", train=True, flatten=True):
        """

        :param mnist_type: "MNIST", "FashionMNIST" or "KMNIST"
        :param train: whether this is train or test
        :param flatten: whether to flatten the data into array or use 2d images
        """
        if mnist_type == "MNIST":
            self.dataset = datasets.MNIST("../../data", train=train, download=True,
                                          transform=torchvision.transforms.Compose([
                                              torchvision.transforms.ToTensor()]))
        elif mnist_type == "FashionMNIST":
            self.dataset = datasets.FashionMNIST(
                "../../data", train=train, download=True, transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor()]))
        elif mnist_type == "KMNIST":
            self.dataset = datasets.KMNIST("../../data", train=train, download=True,
                                           transform=torchvision.transforms.Compose([
                                               torchvision.transforms.ToTensor(),
                                               transforms.Normalize((0.1307,), (0.3081,))]))
        self.transform = transforms.Compose(
            [
                torchvision.transforms.RandomRotation((-45, 45))
            ]
        )
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
        return (x_b, x_a), label


def _add_mnist_noise(x):
    x = x + torch.rand(28, 28) / 10
    return x
