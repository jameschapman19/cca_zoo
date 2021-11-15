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
from cca_zoo.deepmodels import CCALightning, get_dataloaders, architectures, DCCAE, DVCCA

n_train = 500
n_val = 100
train_dataset = Split_MNIST_Dataset(mnist_type='MNIST', train=True)
val_dataset = Subset(train_dataset, np.arange(n_train, n_train + n_val))
train_dataset = Subset(train_dataset, np.arange(n_train))
train_loader, val_loader = get_dataloaders(train_dataset, val_dataset)

# The number of latent dimensions across models
latent_dims = 2
# number of epochs for deep models
epochs = 10

encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=392, variational=True)
encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=392, variational=True)
private_encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=392, variational=True)
private_encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=392, variational=True)
decoder_1 = architectures.Decoder(latent_dims=latent_dims, feature_size=392)
decoder_2 = architectures.Decoder(latent_dims=latent_dims, feature_size=392)
private_decoder_1 = architectures.Decoder(latent_dims=latent_dims, feature_size=392)
private_decoder_2 = architectures.Decoder(latent_dims=latent_dims, feature_size=392)

# Deep VCCA
dcca = DVCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2], decoders=[decoder_1, decoder_2])
dcca = CCALightning(dcca)
trainer = pl.Trainer(max_epochs=epochs, enable_checkpointing=False)
trainer.fit(dcca, train_loader, val_loader)

# Deep VCCA (private)
dcca = DVCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2], decoders=[private_decoder_1, private_decoder_2],
             private_encoders=[private_encoder_1, private_encoder_2])
dcca = CCALightning(dcca)
trainer = pl.Trainer(max_epochs=epochs, enable_checkpointing=False)
trainer.fit(dcca, train_loader, val_loader)

# DCCAE
dccae_model = DCCAE(latent_dims=latent_dims, encoders=[encoder_1, encoder_2], decoders=[decoder_1, decoder_2])
dccae_model = CCALightning(dccae_model)
trainer = pl.Trainer(max_epochs=epochs, enable_checkpointing=False)
trainer.fit(dccae_model, train_loader, val_loader)
