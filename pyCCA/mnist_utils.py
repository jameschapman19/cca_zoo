import gzip

import PIL
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def load_data(data_file):
    """loads the data from the gzip pickled files, and converts to numpy arrays"""
    print('loading data ...')
    f = gzip.open(data_file, 'rb')
    train_set, valid_set, test_set = load_pickle(f)
    f.close()

    train_set_x, train_set_y = make_numpy_array(train_set)
    valid_set_x, valid_set_y = make_numpy_array(valid_set)
    test_set_x, test_set_y = make_numpy_array(test_set)

    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]


def make_numpy_array(data_xy):
    """converts the input to numpy arrays"""
    data_x, data_y = data_xy
    data_x = np.asarray(data_x)
    data_y = np.asarray(data_y, dtype='int32')
    return data_x, data_y


def load_pickle(f):
    """
    loads and returns the content of a pickled file
    it handles the inconsistencies between the pickle packages available in James 2 and 3
    """
    try:
        import cPickle as thepickle
    except ImportError:
        import _pickle as thepickle

    try:
        ret = thepickle.load(f, encoding='latin1')
    except TypeError:
        ret = thepickle.load(f)

    return ret


class Noisy_MNIST_Dataset(Dataset):
    def __init__(self, mnist_type='MNIST', train=True):

        if mnist_type == 'MNIST':
            self.dataset = datasets.MNIST('../../data', train=train, download=True)
        elif mnist_type == 'FashionMNIST':
            self.dataset = datasets.FashionMNIST('../../data', train=train, download=True)
        elif mnist_type == 'KMNIST':
            self.dataset = datasets.KMNIST('../../data', train=train, download=True)

        self.data = self.dataset.data
        self.base_transform = transforms.ToTensor()
        self.a_transform = transforms.Compose([transforms.ToTensor(),  # first, convert image to PyTorch tensor
                                               transforms.ToPILImage()
                                               #                 transforms.Normalize((self.mean,), (self.std,)) # normalize inputs
                                               ])
        self.b_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x + torch.rand(28, 28)),
             transforms.Lambda(lambda x: self.__threshold_func__(x))])
        self.targets = self.dataset.targets
        self.filtered_classes = []
        self.filtered_nums = []
        for i in range(10):
            self.filtered_classes.append(self.data[self.targets == i])
            self.filtered_nums.append(self.filtered_classes[i].shape[0])

    def __threshold_func__(self, x):
        x[x > 1] = 1
        return x

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # get first image from idx and second of same class
        #         x_a = Image.fromarray(self.data[idx].numpy(), mode='L')
        x_a = self.a_transform(self.data[idx].numpy())
        rot_a = torch.rand(1) * 90 - 45
        x_a = transforms.functional.rotate(x_a, rot_a.item(), resample=PIL.Image.BILINEAR)
        x_a = self.base_transform(x_a)  # convert from PIL back to pytorch tensor

        label = self.targets[idx]
        # get random index of image with same class
        random_index = np.random.randint(self.filtered_nums[label])
        x_b = Image.fromarray(self.filtered_classes[label][random_index, :, :].numpy(), mode='L')
        x_b = self.b_transform(x_b)

        return x_a, x_b, rot_a, label


class Tangled_MNIST_Dataset(Dataset):
    def __init__(self, mnist_type='MNIST', train=True, fixed=False):

        if mnist_type == 'MNIST':
            self.dataset = datasets.MNIST('../../data', train=train, download=True)
        elif mnist_type == 'FashionMNIST':
            self.dataset = datasets.FashionMNIST('../../data', train=train, download=True)
        elif mnist_type == 'KMNIST':
            self.dataset = datasets.KMNIST('../../data', train=train, download=True)

        self.data = self.dataset.data
        self.mean = torch.mean(self.data.float())
        self.std = torch.std(self.data.float())
        self.transform = transforms.Compose([transforms.ToTensor(),  # first, convert image to PyTorch tensor
                                             #                     transforms.Lambda(lambda x: x/255.),
                                             #                 transforms.Normalize((self.mean,), (self.std,)) # normalize inputs
                                             ])
        self.targets = self.dataset.targets
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

        return x_a_rotate, x_b_rotate, rot_a, rot_b, label
