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


def plot_reconstruction(model, dataset, idx):
    (x, y), _ = dataset[idx]
    recon_x, recon_y = model.recon(x, y)
    if isinstance(recon_x, list):
        recon_x = recon_x[0]
        recon_y = recon_y[0]
    recon_x = recon_x.detach().numpy()
    recon_y = recon_y.detach().numpy()
    fig, ax = plt.subplots(ncols=4)
    ax[0].set_title('Original View 1')
    ax[1].set_title('Original View 2')
    ax[2].set_title('Reconstruction View 1')
    ax[3].set_title('Reconstruction View 2')
    ax[0].imshow(x.detach().numpy().reshape((28, 28)))
    ax[1].imshow(y.detach().numpy().reshape((28, 28)))
    ax[2].imshow(recon_x.reshape((28, 28)))
    ax[3].imshow(recon_y.reshape((28, 28)))


n_train = 500
n_val = 100
train_dataset = Noisy_MNIST_Dataset(mnist_type="MNIST", train=True, flatten=True)
val_dataset = Subset(train_dataset, np.arange(n_train, n_train + n_val))
train_dataset = Subset(train_dataset, np.arange(n_train))
train_loader, val_loader = get_dataloaders(train_dataset, val_dataset)

# The number of latent dimensions across models
latent_dims = 2
# number of epochs for deep models
epochs = 50

encoder_1 = architectures.Encoder(
    latent_dims=latent_dims, feature_size=784, variational=True
)
encoder_2 = architectures.Encoder(
    latent_dims=latent_dims, feature_size=784, variational=True
)
decoder_1 = architectures.Decoder(latent_dims=latent_dims, feature_size=784, norm_output=True)
decoder_2 = architectures.Decoder(latent_dims=latent_dims, feature_size=784, norm_output=True)

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
private_encoder_1 = architectures.Encoder(
    latent_dims=latent_dims, feature_size=784, variational=True
)
private_encoder_2 = architectures.Encoder(
    latent_dims=latent_dims, feature_size=784, variational=True
)
private_decoder_1 = architectures.Decoder(latent_dims=2 * latent_dims, feature_size=784)
private_decoder_2 = architectures.Decoder(latent_dims=2 * latent_dims, feature_size=784)
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
encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=784)
encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=784)
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
