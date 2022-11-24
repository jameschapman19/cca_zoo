import numpy as np
from multiviewdata.torchdatasets import SplitMNIST, NoisyMNIST
from torch.utils.data import Subset

from cca_zoo.data.deep import get_dataloaders


def example_mnist_data(n_train, n_val, batch_size=50, val_batch_size=10, type="split"):
    if type == "split":
        train_dataset = SplitMNIST(
            root="", mnist_type="MNIST", train=True, download=True
        )
    else:
        train_dataset = NoisyMNIST(
            root="", mnist_type="MNIST", train=True, download=True
        )
    val_dataset = Subset(train_dataset, np.arange(n_train, n_train + n_val))
    train_dataset = Subset(train_dataset, np.arange(n_train))
    train_loader, val_loader = get_dataloaders(
        train_dataset, val_dataset, batch_size=batch_size, val_batch_size=val_batch_size
    )
    train_labels = train_loader.collate_fn(
        [train_dataset.dataset[idx]["label"] for idx in train_dataset.indices]
    ).numpy()
    return train_loader, val_loader, train_labels
