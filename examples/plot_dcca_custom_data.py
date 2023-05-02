"""
Custom Datasets
===========================

This example demonstrates how to use your own multiview datasets with CCA-Zoo.
"""

import numpy as np

# %%
# Imports
# -----
import pytorch_lightning as pl

# %% NumpyDataset
# --------------------------------------------------------------
# This is arguably the easiest way to
# use your own data with CCA-Zoo. You can use the NumpyDataset class to convert your data into a valid dataset. This
# class takes in a list of numpy arrays, each representing a view. It also takes in a list of labels, which can be
# None if you don't have labels. The NumpyDataset class will automatically convert your data into a valid dataset.
# You can then use the get_dataloaders function to convert your dataset into dataloaders, which can be used with
# CCA-Zoo.
from cca_zoo.data.deep import NumpyDataset
from cca_zoo.deepmodels import (
    DCCA,
)
from cca_zoo.deepmodels import architectures

X = np.random.normal(size=(100, 10))
Y = np.random.normal(size=(100, 10))
Z = np.random.normal(size=(100, 10))
numpy_dataset = NumpyDataset([X, Y, Z], labels=None)

# %% Checking the dataset
# ------------------------------------------------------
# You can use the check_dataset function to check if your dataset is valid. This function will check if your
# dataset is a valid CCA-Zoo compatible dataset.
from cca_zoo.data.deep import check_dataset

check_dataset(numpy_dataset)

# %% Your own dataset class
# ------------------------------------------------------
# You can also create your own dataset class. This class must inherit from the torch._utils.data.Dataset class.
# It must also have a __getitem__ method, which returns a tuple of views and a label. The views must be a tuple
# of torch tensors.
import torch


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 10

    def __getitem__(self, index):
        return {"views": (torch.rand(10), torch.rand(10))}


custom_dataset = CustomDataset()
check_dataset(custom_dataset)

# %% Using your own dataset
# ------------------------------------------------------
# You can then use the get_dataloaders function to convert your dataset into dataloaders, which can be used with
# CCA-Zoo.
from cca_zoo.data.deep import get_dataloaders

train_loader = get_dataloaders(custom_dataset, batch_size=2)

# %%
# Training
# -----
LATENT_DIMS = 1
EPOCHS = 10

encoder_1 = architectures.Encoder(latent_dims=LATENT_DIMS, feature_size=10)
encoder_2 = architectures.Encoder(latent_dims=LATENT_DIMS, feature_size=10)

# %%
# Deep CCA
# ----------------------------
dcca = DCCA(latent_dims=LATENT_DIMS, encoders=[encoder_1, encoder_2])
trainer = pl.Trainer(
    max_epochs=EPOCHS,
    enable_checkpointing=False,
    log_every_n_steps=1,
)
trainer.fit(dcca, train_loader)
