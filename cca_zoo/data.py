"""Helped by https://github.com/bcdutton/AdversarialCanonicalCorrelationAnalysis (hopefully I will have my own
implementation of their work soon) Check out their paper at https://arxiv.org/abs/2005.10349 """

import itertools
from typing import List, Union

import PIL
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from scipy import linalg
from scipy.linalg import block_diag
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def generate_covariance_data(n: int, k: int, view_features: List[int],
                             view_sparsity: List[Union[int, float]] = None,
                             signal: float = 1,
                             structure: List[str] = None, sigma: float = 0.9, decay: float = 0.5):
    """
    Function to generate CCA dataset with defined population correlation

    :param view_sparsity: level of sparsity in features in each view either as number of active variables or percentage active
    :param view_features: number of features in each view
    :param n: number of samples
    :param k: number of latent dimensions
    :param signal: correlation
    :param structure: within view covariance structure
    :param sigma: gaussian sigma
    :param decay: ratio of second signal to first signal
    :return: tuple of numpy arrays: view_1, view_2, true weights from view 1, true weights from view 2, overall covariance structure
    """
    if structure is None:
        structure = ['identity'] * len(view_features)
    if view_sparsity is None:
        view_sparsity = [0] * len(view_features)
    completed = False
    while not completed:
        try:
            mean = np.zeros(sum(view_features))
            p = np.arange(0, k)
            p = decay ** p
            cov = []
            true_features = []
            for view_p, sparsity, view_structure in zip(view_features, view_sparsity, structure):
                # Covariance Bit
                if view_structure == 'identity':
                    cov_ = np.eye(view_p)
                elif view_structure == 'gaussian':
                    cov_ = _generate_gaussian_cov(view_p, sigma)
                elif view_structure == 'toeplitz':
                    cov_ = _generate_toeplitz_cov(view_p, sigma)
                elif view_structure == 'random':
                    cov_ = _generate_random_cov(view_p)
                elif view_structure == 'simple':
                    cov_ = generate_simple_data(n, view_features, view_sparsity)
                else:
                    completed = True
                    print("invalid structure")
                weights = np.random.rand(view_p, k)
                if sparsity < 1:
                    sparsity = np.ceil(sparsity * view_p).astype('int')
                mask = np.stack((np.concatenate(([0] * (view_p - sparsity), [1] * sparsity)).astype(bool),) * k,
                                axis=0).T
                np.random.shuffle(mask.flat)
                while np.sum(np.unique(mask, axis=1, return_counts=True)[1] > 1) > 0 or np.sum(
                        np.sum(mask, axis=0) == 0) > 0:
                    np.random.shuffle(mask.flat)
                weights = weights * mask
                weights = _decorrelate_dims(weights, cov_)
                if np.sum(np.diag((weights.T @ cov_ @ weights)) == 0) > 0:
                    print()
                weights /= np.sqrt(np.diag((weights.T @ cov_ @ weights)))
                true_features.append(weights)
                cov.append(cov_)

            cov = block_diag(*cov)

            splits = np.concatenate(([0], np.cumsum(view_features)))

            for i, j in itertools.combinations(range(len(splits) - 1), 2):
                cross = np.zeros((view_features[i], view_features[j]))
                for _ in range(k):
                    cross += signal * p[_] * np.outer(true_features[i][:, _], true_features[j][:, _])
                    # Cross Bit
                    cross = cov[splits[i]:splits[i] + view_features[i], splits[i]:splits[i] + view_features[i]] @ cross \
                            @ cov[splits[j]:splits[j] + view_features[j], splits[j]:splits[j] + view_features[j]]
                cov[splits[i]: splits[i] + view_features[i], splits[j]: splits[j] + view_features[j]] = cross
                cov[splits[j]: splits[j] + view_features[j], splits[i]: splits[i] + view_features[i]] = cross.T

            X = np.zeros((n, sum(view_features)))
            chol = np.linalg.cholesky(cov)
            for _ in range(n):
                X[_, :] = _chol_sample(mean, chol)
            views = np.split(X, np.cumsum(view_features)[:-1], axis=1)
            completed = True
        except:
            completed = False
    return views, true_features


def generate_simple_data(n: int, view_features: List[int], view_sparsity: List[int] = None,
                         eps: float = 0):
    """

    :param n: number of samples
    :param view_features: number of features view 1
    :param view_sparsity: number of features view 2
    :param eps: gaussian noise std
    :return: view1 matrix, view2 matrix, true weights view 1, true weights view 2
    """
    z = np.random.normal(0, 1, n)
    views = []
    true_features = []
    for p, sparsity in zip(view_features, view_sparsity):
        weights = np.random.rand(p, 1)
        if sparsity > 0:
            if sparsity < 1:
                sparsity = np.ceil(sparsity * p).astype('int')
            weights[np.random.choice(np.arange(p), p - sparsity, replace=False)] = 0

        gaussian_x = np.random.normal(0, eps, (n, p))
        view = np.outer(z, weights)
        view += gaussian_x
        views.append(view)
        true_features.append(weights)
    return views, true_features


class CCA_Dataset(Dataset):
    """
    Class that turns numpy arrays into a torch dataset

    """

    def __init__(self, *views, labels=None, train=True):
        """

        :param views:
        :param labels:
        :param train:
        """
        self.train = train
        self.views = [view for view in views]
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
        OH_labels = _OH_digits(labels.astype(int))
        return view_1, view_2, rotations, OH_labels, labels


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
        OH_labels = _OH_digits(labels.astype(int))
        return view_1, view_2, rotation_1, rotation_2, OH_labels, labels


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


def _decorrelate_dims(up, cov):
    A = up.T @ cov @ up
    for k in range(1, A.shape[0]):
        up[:, k:] -= np.outer(up[:, k - 1], A[k - 1, k:] / A[k - 1, k - 1])
        A = up.T @ cov @ up
    return up


def _chol_sample(mean, chol):
    return mean + chol @ np.random.standard_normal(mean.size)


def _gaussian(x, mu, sig, dn):
    """
    Generate a gaussian covariance matrix

    :param x:
    :param mu:
    :param sig:
    :param dn:
    """
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) * dn / (np.sqrt(2 * np.pi) * sig)


def _generate_gaussian_cov(p, sigma):
    x = np.linspace(-1, 1, p)
    x_tile = np.tile(x, (p, 1))
    mu_tile = np.transpose(x_tile)
    dn = 2 / (p - 1)
    cov = _gaussian(x_tile, mu_tile, sigma, dn)
    cov /= cov.max()
    return cov


def _generate_toeplitz_cov(p, sigma):
    c = np.arange(0, p)
    c = sigma ** c
    cov = linalg.toeplitz(c, c)
    return cov


def _generate_random_cov(p):
    cov_ = np.random.rand(p, p)
    U, S, V = np.linalg.svd(cov_.T @ cov_)
    cov = U @ (1.0 + np.diag(np.random.rand(p))) @ V
    return cov
