"""Helped by https://github.com/bcdutton/AdversarialCanonicalCorrelationAnalysis (hopefully I will have my own
implementation of their work soon) Check out their paper at https://arxiv.org/abs/2005.10349 """

import numpy as np
import torch
import torch.utils.data
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode


class Split_MNIST_Dataset(Dataset):
    """
    Class to generate paired noisy mnist data
    """

    def __init__(self, mnist_type: str = 'MNIST', train: bool = True, flatten: bool = True):
        """

        :param mnist_type:
        :param train:
        :param flatten:
        """
        if mnist_type == 'MNIST':
            self.dataset = datasets.MNIST('../../data', train=train, download=True)
        elif mnist_type == 'FashionMNIST':
            self.dataset = datasets.FashionMNIST('../../data', train=train, download=True)
        elif mnist_type == 'KMNIST':
            self.dataset = datasets.KMNIST('../../data', train=train, download=True)

        self.data = self.dataset.data
        self.base_transform = transforms.ToTensor()
        self.targets = self.dataset.targets
        self.flatten = flatten

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx].flatten()
        x_a = x[:392]
        x_b = x[392:]
        label = self.targets[idx]
        return (x_a, x_b), label

    def to_numpy(self, indices=None):
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

    def __init__(self, mnist_type: str = 'MNIST', train: bool = True, flatten: bool = True):
        """

        :param mnist_type:
        :param train:
        :param flatten:
        """
        if mnist_type == 'MNIST':
            self.dataset = datasets.MNIST('../../data', train=train, download=True)
        elif mnist_type == 'FashionMNIST':
            self.dataset = datasets.FashionMNIST('../../data', train=train, download=True)
        elif mnist_type == 'KMNIST':
            self.dataset = datasets.KMNIST('../../data', train=train, download=True)

        self.data = self.dataset.data
        self.base_transform = transforms.ToTensor()
        self.a_transform = transforms.Compose([transforms.ToTensor(),  # first, convert image to PyTorch tensor
                                               transforms.ToPILImage()])
        self.b_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(_add_mnist_noise),
             transforms.Lambda(self.__threshold_func__)])
        self.targets = self.dataset.targets
        self.OHs = _OH_digits(self.targets.numpy().astype(int))
        self.filtered_classes = []
        self.filtered_nums = []
        for i in range(10):
            self.filtered_classes.append(self.data[self.targets == i])
            self.filtered_nums.append(self.filtered_classes[i].shape[0])
        self.flatten = flatten

    def __threshold_func__(self, x):
        x[x > 1] = 1
        return x

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_a = self.a_transform(self.data[idx].numpy())
        rot_a = torch.rand(1) * 90 - 45
        x_a = transforms.functional.rotate(x_a, rot_a.item(), interpolation=InterpolationMode.BILINEAR)
        x_a = self.base_transform(x_a)  # convert from PIL back to pytorch tensor

        label = self.targets[idx]
        # get random index of image with same class
        random_index = np.random.randint(self.filtered_nums[label])
        x_b = Image.fromarray(self.filtered_classes[label][random_index, :, :].numpy(), mode='L')
        x_b = self.b_transform(x_b)

        if self.flatten:
            x_a = torch.flatten(x_a)
            x_b = torch.flatten(x_b)
        return (x_b, x_a), (rot_a, label)

    def to_numpy(self, indices=None):
        if indices is None:
            indices = np.arange(self.__len__())
        view_1 = np.zeros((len(indices), 784))
        view_2 = np.zeros((len(indices), 784))
        labels = np.zeros(len(indices)).astype(int)
        rotations = np.zeros(len(indices))
        for i, n in enumerate(indices):
            sample = self[n]
            view_1[i] = sample[0][0].numpy().reshape((-1, 28 * 28))
            view_2[i] = sample[0][1].numpy().reshape((-1, 28 * 28))
            rotations[i] = sample[1][0].numpy()
            labels[i] = sample[1][1].numpy().astype(int)
        return (view_1, view_2), (rotations, labels)


class Tangled_MNIST_Dataset(Dataset):
    """
    Class to generate paired tangled MNIST dataset
    """

    def __init__(self, mnist_type='MNIST', train=True, fixed=False, flatten=True):
        """

        :param mnist_type:
        :param train:
        :param fixed:
        :param flatten:
        """
        if mnist_type == 'MNIST':
            self.dataset = datasets.MNIST('../../data', train=train, download=True)
        elif mnist_type == 'FashionMNIST':
            self.dataset = datasets.FashionMNIST('../../data', train=train, download=True)
        elif mnist_type == 'KMNIST':
            self.dataset = datasets.KMNIST('../../data', train=train, download=True)

        self.data = self.dataset.data
        self.mean = torch.mean(self.data.float())
        self.std = torch.std(self.data.float())
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.targets = self.dataset.targets
        self.OHs = _OH_digits(self.targets.numpy().astype(int))
        self.fixed = fixed
        self.filtered_classes = []
        self.filtered_nums = []
        for i in range(10):
            self.filtered_classes.append(self.data[self.targets == i])
            self.filtered_nums.append(self.filtered_classes[i].shape[0])
        if fixed:
            self.view_b_indices = []
            for i in range(10):
                self.view_b_indices.append(np.random.permutation(np.arange(len(self.data))[self.targets == i]))

        self.flatten = flatten

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # get first image from idx and second of same class
        label = self.targets[idx]
        x_a = Image.fromarray(self.data[idx].numpy(), mode='L')
        # get random index of image with same class
        random_index = np.random.randint(self.filtered_nums[label])
        x_b = Image.fromarray(self.filtered_classes[label][random_index, :, :].numpy(), mode='L')
        # get random angles of rotation
        rot_a, rot_b = torch.rand(2) * 90 - 45
        x_a_rotate = transforms.functional.rotate(x_a, rot_a.item(), interpolation=InterpolationMode.BILINEAR)
        x_b_rotate = transforms.functional.rotate(x_b, rot_b.item(), interpolation=InterpolationMode.BILINEAR)
        # convert images to tensors
        x_a_rotate = self.transform(x_a_rotate)
        x_b_rotate = self.transform(x_b_rotate)

        if self.flatten:
            x_a_rotate = torch.flatten(x_a_rotate)
            x_b_rotate = torch.flatten(x_b_rotate)
        return (x_a_rotate, x_b_rotate), (rot_a, rot_b, label)

    def to_numpy(self, indices):
        view_1 = np.zeros((len(indices), 784))
        view_2 = np.zeros((len(indices), 784))
        labels = np.zeros(len(indices)).astype(int)
        rotation_1 = np.zeros(len(indices))
        rotation_2 = np.zeros(len(indices))
        for i, n in enumerate(indices):
            sample = self[n]
            view_1[i] = sample[0][0].numpy().reshape((-1, 28 * 28))
            view_2[i] = sample[0][1].numpy().reshape((-1, 28 * 28))
            rotation_1[i] = sample[1][0].numpy()
            rotation_2[i] = sample[1][1].numpy()
            labels[i] = sample[1][2].numpy().astype(int)
        return (view_1, view_2), (rotation_1, rotation_2, labels)


def _OH_digits(digits):
    """
    One hot encode numpy array

    :param digits:
    """
    b = np.zeros((digits.size, digits.max() + 1))
    b[np.arange(digits.size), digits] = 1
    return b


def _add_mnist_noise(x):
    x = x + torch.rand(28, 28)
    return x
