"""
Working with Custom Datasets in CCA-Zoo
=======================================

This script provides a guide on how to leverage custom multiview datasets with
CCA-Zoo. It walks through various methods, including the use of provided
utilities and the creation of a bespoke dataset class.

Key Features:
- Transforming numpy arrays into CCA-Zoo compatible datasets.
- Validating custom datasets.
- Creating a custom dataset class from scratch.
- Training a Deep CCA model on custom datasets.
"""

import numpy as np
import pytorch_lightning as pl

# %%
# Converting Numpy Arrays into Datasets
# -------------------------------------
# For those looking for a straightforward method, the `NumpyDataset` class from CCA-Zoo
# is a convenient way to convert numpy arrays into valid datasets. It accepts multiple
# numpy arrays, each representing a distinct view, and an optional list of labels.
# Subsequently, these datasets can be converted into dataloaders for use in CCA-Zoo models.

from cca_zoo.data.deep import NumpyDataset
from cca_zoo.deep import DCCA, architectures

X = np.random.normal(size=(100, 10))
Y = np.random.normal(size=(100, 10))
Z = np.random.normal(size=(100, 10))

numpy_dataset = NumpyDataset([X, Y, Z], labels=None)

# %%
# Dataset Validation
# ------------------
# Before proceeding, it's a good practice to validate the constructed dataset.
# The `check_dataset` function ensures that the dataset adheres to CCA-Zoo's
# expected format.

from cca_zoo.data.deep import check_dataset

check_dataset(numpy_dataset)

# %%
# Creating a Custom Dataset Class
# -------------------------------
# For advanced users or specific requirements, one can create a custom dataset class.
# The new class should inherit from the native `torch.utils.data.Dataset` class.
# The class must implement the `__getitem__` method to return a tuple consisting
# of multiple views and an associated label, where views are represented as torch tensors.

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

# %%
# Convert Custom Dataset into DataLoader
# --------------------------------------
# The `get_dataloaders` function can now be used to transform the custom dataset
# into dataloaders suitable for CCA-Zoo.

from cca_zoo.data.deep import get_dataloaders

train_loader = get_dataloaders(custom_dataset, batch_size=2)

# %%
# Training with Deep CCA
# -----------------------
# Once the dataloaders are set, it's time to configure and train a Deep CCA model.

LATENT_DIMS = 1
EPOCHS = 10

encoder_1 = architectures.Encoder(latent_dimensions=LATENT_DIMS, feature_size=10)
encoder_2 = architectures.Encoder(latent_dimensions=LATENT_DIMS, feature_size=10)

dcca = DCCA(latent_dimensions=LATENT_DIMS, encoders=[encoder_1, encoder_2])
trainer = pl.Trainer(
    max_epochs=EPOCHS,
    enable_checkpointing=False,
    log_every_n_steps=1,
)
trainer.fit(dcca, train_loader)
