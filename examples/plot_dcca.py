"""
Deep CCA
===========================

This example demonstrates how to easily train Deep CCA models and variants
"""

import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from torch.utils.data import Subset

# %%
from multiviewdata.torchdatasets import SplitMNIST
from cca_zoo.deepmodels import (
    DCCA,
    get_dataloaders,
    architectures,
    DCCA_NOI,
    DCCA_SDL,
    BarlowTwins,
)

n_train = 500
n_val = 100
train_dataset = SplitMNIST(root="", mnist_type="MNIST", train=True, download=True)
val_dataset = Subset(train_dataset, np.arange(n_train, n_train + n_val))
train_dataset = Subset(train_dataset, np.arange(n_train))
train_loader, val_loader = get_dataloaders(train_dataset, val_dataset, batch_size=50)

# The number of latent dimensions across models
latent_dims = 2
# number of epochs for deep models
epochs = 50

encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=392)
encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=392)

# %%
# Deep CCA
dcca = DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2])
trainer = pl.Trainer(
    max_epochs=epochs,
    enable_checkpointing=False,
    log_every_n_steps=1,
    flush_logs_every_n_steps=1,
)
trainer.fit(dcca, train_loader, val_loader)
dcca.plot_latent_label(train_loader)
plt.suptitle("DCCA")
plt.show()

# %%
# Deep CCA by Non-Linear Orthogonal Iterations
dcca_noi = DCCA_NOI(
    latent_dims=latent_dims, N=len(train_dataset), encoders=[encoder_1, encoder_2]
)
trainer = pl.Trainer(
    max_epochs=epochs,
    enable_checkpointing=False,
    log_every_n_steps=1,
    flush_logs_every_n_steps=1,
)
trainer.fit(dcca_noi, train_loader, val_loader)
dcca_noi.plot_latent_label(train_loader)
plt.suptitle("DCCA by Non-Linear Orthogonal Iterations")
plt.show()

# %%
# Deep CCA by Stochastic Decorrelation Loss
dcca_sdl = DCCA_SDL(
    latent_dims=latent_dims, N=len(train_dataset), encoders=[encoder_1, encoder_2]
)
trainer = pl.Trainer(
    max_epochs=epochs,
    enable_checkpointing=False,
    log_every_n_steps=1,
    flush_logs_every_n_steps=1,
)
trainer.fit(dcca_sdl, train_loader, val_loader)
dcca_sdl.plot_latent_label(train_loader)
plt.suptitle("DCCA by Stochastic Decorrelation")
plt.show()

# %%
# Deep CCA by Barlow Twins
barlowtwins = BarlowTwins(latent_dims=latent_dims, encoders=[encoder_1, encoder_2])
trainer = pl.Trainer(
    max_epochs=epochs,
    enable_checkpointing=False,
    log_every_n_steps=1,
    flush_logs_every_n_steps=1,
)
trainer.fit(barlowtwins, train_loader, val_loader)
barlowtwins.plot_latent_label(train_loader)
plt.suptitle("DCCA by Barlow Twins")
plt.show()
