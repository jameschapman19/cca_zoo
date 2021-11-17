"""
Deep CCA with more customisation
==================================

Showing some examples of more advanced functionality with DCCA and pytorch-lightning
"""

import numpy as np
# %%
import pytorch_lightning as pl
from torch import optim
from torch.utils.data import Subset

from cca_zoo.data import Split_MNIST_Dataset
from cca_zoo.deepmodels import DCCA, CCALightning, get_dataloaders, architectures

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

# TODO add in custom architecture and schedulers and stuff to show it off
encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=392)
encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=392)

# Deep CCA
dcca = DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2])
optimizer = optim.Adam(dcca.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 1)
dcca = CCALightning(dcca, optimizer=optimizer, lr_scheduler=scheduler)
trainer = pl.Trainer(max_epochs=epochs, enable_checkpointing=False)
trainer.fit(dcca, train_loader, val_loader)
