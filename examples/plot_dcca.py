"""
Deep CCA
===========================

This example demonstrates how to easily train Deep CCA models and variants
"""

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Subset

# %%
from cca_zoo.data import Split_MNIST_Dataset
from cca_zoo.deepmodels import (
    DCCA,
    CCALightning,
    get_dataloaders,
    architectures,
    DCCA_NOI,
    DCCA_SDL,
    BarlowTwins,
)

n_train = 500
n_val = 100
train_dataset = Split_MNIST_Dataset(mnist_type="MNIST", train=True)
val_dataset = Subset(train_dataset, np.arange(n_train, n_train + n_val))
train_dataset = Subset(train_dataset, np.arange(n_train))
train_loader, val_loader = get_dataloaders(train_dataset, val_dataset)

# The number of latent dimensions across models
latent_dims = 2
# number of epochs for deep models
epochs = 10

encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=392)
encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=392)

# Deep CCA
dcca = DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2])
dcca = CCALightning(dcca)
trainer = pl.Trainer(max_epochs=epochs, enable_checkpointing=False)
trainer.fit(dcca, train_loader, val_loader)

# Deep CCA by Non-Linear Orthogonal Iterations
dcca_noi = DCCA_NOI(
    latent_dims=latent_dims, N=len(train_dataset), encoders=[encoder_1, encoder_2]
)
dcca_noi = CCALightning(dcca_noi)
trainer = pl.Trainer(max_epochs=epochs, enable_checkpointing=False)
trainer.fit(dcca_noi, train_loader, val_loader)

# Deep CCA by Stochastic Decorrelation Loss
dcca_sdl = DCCA_SDL(
    latent_dims=latent_dims, N=len(train_dataset), encoders=[encoder_1, encoder_2]
)
dcca_sdl = CCALightning(dcca_sdl)
trainer = pl.Trainer(max_epochs=epochs, enable_checkpointing=False)
trainer.fit(dcca_sdl, train_loader, val_loader)

# Deep CCA by Barlow Twins
barlowtwins = BarlowTwins(latent_dims=latent_dims, encoders=[encoder_1, encoder_2])
barlowtwins = CCALightning(barlowtwins)
trainer = pl.Trainer(max_epochs=epochs, enable_checkpointing=False)
trainer.fit(dcca, train_loader, val_loader)