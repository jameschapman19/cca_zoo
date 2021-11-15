"""
Deep CCA with more customisation
==================================

Showing some examples of more advanced functionality
"""

# %%
import pytorch_lightning as pl

from cca_zoo.data import Split_MNIST_Dataset
from cca_zoo.deepmodels import DCCA, CCALightning, get_dataloaders, architectures

train_dataset = Split_MNIST_Dataset(mnist_type='FashionMNIST', train=True)
test_dataset = Split_MNIST_Dataset(mnist_type='FashionMNIST', train=False)
train_loader, val_loader = get_dataloaders(train_dataset, test_dataset)

# The number of latent dimensions across models
latent_dims = 2
# number of epochs for deep models
epochs = 10

# TODO
encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=392)
encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=392)

# Deep CCA
dcca = DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2])
dcca = CCALightning(dcca)
trainer = pl.Trainer(max_epochs=epochs, enable_checkpointing=False)
trainer.fit(dcca, train_loader, val_loader)
