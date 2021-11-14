"""
Deep CCA
===========================

This example demonstrates how to easily train Deep CCA models and variants
"""

import pytorch_lightning as pl

# %%
from cca_zoo.data import Noisy_MNIST_Dataset
from cca_zoo.deepmodels import CCALightning, get_dataloaders, architectures, DCCAE, DVCCA

train_dataset = Noisy_MNIST_Dataset(mnist_type='FashionMNIST', train=True)
test_dataset = Noisy_MNIST_Dataset(mnist_type='FashionMNIST', train=False)
train_loader, val_loader = get_dataloaders(train_dataset, test_dataset)

# The number of latent dimensions across models
latent_dims = 2
# number of epochs for deep models
epochs = 10

encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=784)
encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=784)
private_encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=784, variational=True)
private_encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=784, variational=True)
decoder_1 = architectures.Decoder(latent_dims=latent_dims, feature_size=784)
decoder_2 = architectures.Decoder(latent_dims=latent_dims, feature_size=784)

# DCCAE
dccae_model = DCCAE(latent_dims=latent_dims, encoders=[encoder_1, encoder_2], decoders=[decoder_1, decoder_2])
dccae_model = CCALightning(dccae_model)
trainer = pl.Trainer(max_epochs=epochs, progress_bar_refresh_rate=1, log_every_n_steps=1, logger=False)
trainer.fit(dccae_model, train_loader, val_loader)

# Deep VCCA
dcca = DVCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2], decoders=[decoder_1, decoder_2])
dcca = CCALightning(dcca)
trainer = pl.Trainer(max_epochs=epochs, progress_bar_refresh_rate=1, log_every_n_steps=1, logger=False)
trainer.fit(dcca, train_loader, val_loader)

# Deep VCCA (private)
dcca = DVCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2], decoders=[decoder_1, decoder_2],
             private_encoders=[private_encoder_1, private_encoder_2])
dcca = CCALightning(dcca)
trainer = pl.Trainer(max_epochs=epochs, progress_bar_refresh_rate=1, log_every_n_steps=1, logger=False)
trainer.fit(dcca, train_loader, val_loader)
