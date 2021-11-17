"""
Deep Variational CCA and Deep Canonically Correlated Autoencoders
====================================================================

This example demonstrates multiview models which can reconstruct their inputs
"""

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Subset

# %%
from cca_zoo.data import Noisy_MNIST_Dataset
from cca_zoo.deepmodels import (
    CCALightning,
    get_dataloaders,
    architectures,
    DCCAE,
    DVCCA,
)
from examples.utils import plot_reconstruction

n_train = 500
n_val = 100
train_dataset = Noisy_MNIST_Dataset(mnist_type="MNIST", train=True, flatten=False)
val_dataset = Subset(train_dataset, np.arange(n_train, n_train + n_val))
train_dataset = Subset(train_dataset, np.arange(n_train))
train_loader, val_loader = get_dataloaders(train_dataset, val_dataset)

# The number of latent dimensions across models
latent_dims = 2
# number of epochs for deep models
epochs = 10
# channels in encoders and decoders
channels = [16, 16]

encoder_1 = architectures.CNNEncoder(
    latent_dims=latent_dims, feature_size=(28, 28), variational=True, channels=channels,
)
encoder_2 = architectures.CNNEncoder(
    latent_dims=latent_dims, feature_size=(28, 28), variational=True, channels=channels,
)
decoder_1 = architectures.CNNDecoder(latent_dims=latent_dims, feature_size=(28, 28))
decoder_2 = architectures.CNNDecoder(latent_dims=latent_dims, feature_size=(28, 28))

# %%
# Deep VCCA
dcca = DVCCA(
    latent_dims=latent_dims,
    encoders=[encoder_1, encoder_2],
    decoders=[decoder_1, decoder_2],
)
dcca = CCALightning(dcca)
trainer = pl.Trainer(max_epochs=epochs, enable_checkpointing=False)
trainer.fit(dcca, train_loader, val_loader)
plot_reconstruction(dcca.model, train_dataset, 0)
plt.suptitle('DVCCA')
plt.show()

# %%
# Deep VCCA (private)
# We need to add additional private encoders and change (double) the dimensionality of the decoders.
private_encoder_1 = architectures.CNNEncoder(
    latent_dims=latent_dims, feature_size=(28, 28), variational=True, channels=channels,
)
private_encoder_2 = architectures.CNNEncoder(
    latent_dims=latent_dims, feature_size=(28, 28), variational=True, channels=channels,
)
private_decoder_1 = architectures.CNNDecoder(latent_dims=2 * latent_dims, feature_size=(28, 28))
private_decoder_2 = architectures.CNNDecoder(latent_dims=2 * latent_dims, feature_size=(28, 28))
dcca = DVCCA(
    latent_dims=latent_dims,
    encoders=[encoder_1, encoder_2],
    decoders=[private_decoder_1, private_decoder_2],
    private_encoders=[private_encoder_1, private_encoder_2],
)
dcca = CCALightning(dcca)
trainer = pl.Trainer(max_epochs=epochs, enable_checkpointing=False)
trainer.fit(dcca, train_loader, val_loader)
plot_reconstruction(dcca.model, train_dataset, 0)
plt.suptitle('DVCCA Private')
plt.show()

# %%
# DCCAE
encoder_1 = architectures.CNNEncoder(latent_dims=latent_dims, feature_size=(28, 28), channels=channels, )
encoder_2 = architectures.CNNEncoder(latent_dims=latent_dims, feature_size=(28, 28), channels=channels, )
dcca = DCCAE(
    latent_dims=latent_dims,
    encoders=[encoder_1, encoder_2],
    decoders=[decoder_1, decoder_2],
)
dcca = CCALightning(dcca)
trainer = pl.Trainer(max_epochs=epochs, enable_checkpointing=False)
trainer.fit(dcca, train_loader, val_loader)
plot_reconstruction(dcca.model, train_dataset, 0)
plt.suptitle('DCCAE')
plt.show()
