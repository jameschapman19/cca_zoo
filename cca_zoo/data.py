"""
Helped by https://github.com/bcdutton/AdversarialCanonicalCorrelationAnalysis (hopefully I will have my own implementation of their work soon)
Check out their paper at https://arxiv.org/abs/2005.10349
"""

import PIL
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from mvlearn.datasets import load_UCImultifeature
from scipy import linalg
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def OH_digits(digits):
    b = np.zeros((digits.size, digits.max() + 1))
    b[np.arange(digits.size), digits] = 1
    return b


def add_mnist_noise(x):
    x = x + torch.rand(28, 28)
    return x


class CCA_Dataset(Dataset):
    def __init__(self, *args, labels=None, train=True):
        self.train = train
        self.views = [view for view in args]
        if labels is None:
            self.labels = np.zeros(len(self.views[0]))
        else:
            self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        views = [view[idx] for view in self.views]
        return tuple(views), label


class Noisy_MNIST_Dataset(Dataset):
    def __init__(self, mnist_type='MNIST', train=True, flatten=True):
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
            [transforms.ToTensor(), transforms.Lambda(add_mnist_noise),
             transforms.Lambda(self.__threshold_func__)])
        self.targets = self.dataset.targets
        self.OHs = OH_digits(self.targets.numpy().astype(int))
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
        x_a = transforms.functional.rotate(x_a, rot_a.item(), resample=PIL.Image.BILINEAR)
        x_a = self.base_transform(x_a)  # convert from PIL back to pytorch tensor

        label = self.targets[idx]
        # get random index of image with same class
        random_index = np.random.randint(self.filtered_nums[label])
        x_b = Image.fromarray(self.filtered_classes[label][random_index, :, :].numpy(), mode='L')
        x_b = self.b_transform(x_b)

        if self.flatten:
            x_a = torch.flatten(x_a)
            x_b = torch.flatten(x_b)
        OH = self.OHs[idx]
        return (x_b, x_a, rot_a, OH), label

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
            rotations[i] = sample[0][2].numpy()
            labels[i] = sample[1].numpy().astype(int)
        OH_labels = OH_digits(labels.astype(int))
        return view_1, view_2, rotations, OH_labels, labels


class Tangled_MNIST_Dataset(Dataset):
    def __init__(self, mnist_type='MNIST', train=True, fixed=False, flatten=True):

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
        self.OHs = OH_digits(self.targets.numpy().astype(int))
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
        x_a_rotate = transforms.functional.rotate(x_a, rot_a.item(), resample=PIL.Image.BILINEAR)
        x_b_rotate = transforms.functional.rotate(x_b, rot_b.item(), resample=PIL.Image.BILINEAR)
        # convert images to tensors
        x_a_rotate = self.transform(x_a_rotate)
        x_b_rotate = self.transform(x_b_rotate)

        if self.flatten:
            x_a_rotate = torch.flatten(x_a_rotate)
            x_b_rotate = torch.flatten(x_b_rotate)
        OH = self.OHs[idx]
        return (x_a_rotate, x_b_rotate, rot_a, rot_b, OH), label

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
            rotation_1[i] = sample[0][2].numpy()
            rotation_2[i] = sample[0][3].numpy()
            labels[i] = sample[1].numpy().astype(int)
        OH_labels = OH_digits(labels.astype(int))
        return view_1, view_2, rotation_1, rotation_2, OH_labels, labels


# Copyright (c) 2020 The mvlearn developers.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

class UCI_Dataset(Dataset):
    def __init__(self, train=True):
        full_data, self.labels = load_UCImultifeature()
        self.train = train
        self.view_1, self.view_2, self.view_3, self.view_4, self.view_5, self.view_6 = full_data
        self.view_1 = MinMaxScaler().fit_transform(self.view_1)
        self.view_2 = MinMaxScaler().fit_transform(self.view_2)
        self.view_3 = MinMaxScaler().fit_transform(self.view_3)
        self.view_4 = MinMaxScaler().fit_transform(self.view_4)
        self.view_5 = MinMaxScaler().fit_transform(self.view_5)
        self.view_6 = MinMaxScaler().fit_transform(self.view_6)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        view_1 = self.view_1[idx]
        view_2 = self.view_2[idx]
        view_3 = self.view_3[idx]
        view_4 = self.view_4[idx]
        view_5 = self.view_5[idx]
        view_6 = self.view_6[idx]
        return (view_1, view_2, view_3, view_4, view_5, view_6), label

    def to_numpy(self, indices):
        labels = self.labels[indices]
        view_1 = self.view_1[indices]
        view_2 = self.view_2[indices]
        view_3 = self.view_3[indices]
        view_4 = self.view_4[indices]
        view_5 = self.view_5[indices]
        view_6 = self.view_6[indices]
        OH_labels = OH_digits(labels.astype(int))
        return view_1, view_2, view_3, view_4, view_5, view_6, OH_labels, labels


def chol_sample(mean, chol):
    return mean + chol @ np.random.standard_normal(mean.size)


def gaussian(x, mu, sig, dn):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) * dn / (np.sqrt(2 * np.pi) * sig)


def generate_simulated_data(m: int, k: int, N: int, M: int, sparse_variables_1: float = 0,
                            sparse_variables_2: float = 0,
                            signal: float = 1,
                            structure: str = 'identity', sigma: float = 0.9, decay: float = 0.5,
                            rand_eigs_1: bool = False,
                            rand_eigs_2: bool = False):
    """
    :param m: number of samples
    :param k: number of latent dimensions
    :param N: number of features in view 1
    :param M: number of features in view 2
    :param sparse_variables_1: fraction of active variables from view 1 associated with true signal
    :param sparse_variables_2: fraction of active variables from view 2 associated with true signal
    :param signal: correlation
    :param structure: within view covariance structure
    :param sigma: gaussian sigma
    :param decay: ratio of second signal to first signal
    :param rand_eigs_1:
    :param rand_eigs_2:
    :return: tuple of numpy arrays: view_1, view_2, true weights from view 1, true weights from view 2, overall covariance structure
    """
    mean = np.zeros(N + M)
    cov = np.zeros((N + M, N + M))
    p = np.arange(0, k)
    p = decay ** p
    # Covariance Bit
    if structure == 'identity':
        cov_1 = np.eye(N)
        cov_2 = np.eye(M)
    elif structure == 'gaussian':
        x = np.linspace(-1, 1, N)
        x_tile = np.tile(x, (N, 1))
        mu_tile = np.transpose(x_tile)
        dn = 2 / (N - 1)
        cov_1 = gaussian(x_tile, mu_tile, sigma, dn)
        cov_1 /= cov_1.max()
        x = np.linspace(-1, 1, M)
        x_tile = np.tile(x, (M, 1))
        mu_tile = np.transpose(x_tile)
        dn = 2 / (M - 1)
        cov_2 = gaussian(x_tile, mu_tile, sigma, dn)
        cov_2 /= cov_2.max()
    elif structure == 'toeplitz':
        c = np.arange(0, N)
        c = sigma ** c
        cov_1 = linalg.toeplitz(c, c)
        c = np.arange(0, M)
        c = sigma ** c
        cov_2 = linalg.toeplitz(c, c)
    elif structure == 'random':
        cov_1 = np.random.rand(N, N)
        U, S, V = np.linalg.svd(cov_1.T @ cov_1)
        cov_1 = U @ (1.0 + np.diag(np.random.rand(N))) @ V
        cov_2 = np.random.rand(M, M)
        U, S, V = np.linalg.svd(cov_2.T @ cov_2)
        cov_2 = U @ (1.0 + np.diag(np.random.rand(M))) @ V
    cov[:N, :N] = cov_1
    cov[N:, N:] = cov_2
    del cov_1
    del cov_2

    up = np.random.rand(N, k) - 0.5
    for _ in range(k):
        if sparse_variables_1 > 0:
            if sparse_variables_1 < 1:
                sparse_variables_1 = np.ceil(sparse_variables_1 * N).astype('int')
            first = np.random.randint(N - sparse_variables_1)
            up[:first, _] = 0
            up[(first + sparse_variables_1):, _] = 0

    up = decorrelate_dims(up, cov[:N, :N])
    up /= np.sqrt(np.diag((up.T @ cov[:N, :N] @ up)))

    vp = np.random.rand(M, k) - 0.5
    for _ in range(k):
        if sparse_variables_2 > 0:
            if sparse_variables_2 < 1:
                sparse_variables_2 = np.ceil(sparse_variables_2 * M).astype('int')
            first = np.random.randint(M - sparse_variables_2)
            vp[:first, _] = 0
            vp[(first + sparse_variables_2):, _] = 0

    vp = decorrelate_dims(vp, cov[N:, N:])
    vp /= np.sqrt(np.diag((vp.T @ cov[N:, N:] @ vp)))

    cross = np.zeros((N, M))
    for _ in range(k):
        cross += signal * p[_] * np.outer(up[:, _], vp[:, _])
    # Cross Bit
    cross = cov[:N, :N] @ cross @ cov[N:, N:]

    cov[N:, :N] = cross.T
    cov[:N, N:] = cross

    if cov.shape[0] < 2000:
        X = np.random.multivariate_normal(mean, cov, m)
    else:
        X = np.zeros((m, N + M))
        chol = np.linalg.cholesky(cov)
        for _ in range(m):
            X[_, :] = chol_sample(mean, chol)
    Y = X[:, N:]
    X = X[:, :N]
    return X, Y, up, vp, cov


def decorrelate_dims(up, cov):
    A = up.T @ cov @ up
    for k in range(1, A.shape[0]):
        up[:, k:] -= np.outer(up[:, k-1], A[k-1, k:]/A[k-1, k-1])
        A = up.T @ cov @ up
    return up
