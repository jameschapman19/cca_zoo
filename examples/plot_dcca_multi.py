"""
Deep CCA for more than 2 views
=================================

This example demonstrates how to easily train Deep CCA models and variants
"""

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Subset

# %%
from multiviewdata.torchdatasets import SplitMNIST
from cca_zoo.deepmodels import (
    DCCA,
    get_dataloaders,
    architectures,
    objectives,
    DTCCA,
)

n_train = 500
n_val = 100
train_dataset = SplitMNIST(root="", mnist_type="MNIST", train=True, download=True)
val_dataset = Subset(train_dataset, np.arange(n_train, n_train + n_val))
train_dataset = Subset(train_dataset, np.arange(n_train))
train_loader, val_loader = get_dataloaders(train_dataset, val_dataset)

# The number of latent dimensions across models
latent_dims = 2
# number of epochs for deep models
epochs = 10

encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=392)
encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=392)

# %%
# Deep MCCA
dcca = DCCA(
    latent_dims=latent_dims, encoders=[encoder_1, encoder_2], objective=objectives.MCCA
)
trainer = pl.Trainer(max_epochs=epochs, enable_checkpointing=False)
trainer.fit(dcca, train_loader, val_loader)

# %%
# Deep GCCA
dcca = DCCA(
    latent_dims=latent_dims, encoders=[encoder_1, encoder_2], objective=objectives.GCCA
)
trainer = pl.Trainer(max_epochs=epochs, enable_checkpointing=False)
trainer.fit(dcca, train_loader, val_loader)

# %%
# Deep TCCA
dcca = DTCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2])
trainer = pl.Trainer(max_epochs=epochs, enable_checkpointing=False)
trainer.fit(dcca, train_loader, val_loader)
