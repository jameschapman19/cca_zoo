"""
Deep Variational CCA and Deep Canonically Correlated Autoencoders
====================================================================

This example demonstrates multiview models which can reconstruct their inputs
"""
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from multiviewdata.torchdatasets import NoisyMNIST

# %%
from torch.utils.data import Subset

from cca_zoo.deepmodels import get_dataloaders, architectures, DCCAE, DVCCA, SplitAE


def plot_reconstruction(model, dataloader):
    for i, batch in enumerate(dataloader):
        x, y = batch["views"]
        output = model(x[0], y[0], mle=True)
        if isinstance(output, tuple):
            z = output[0]
        else:
            z = output
    recons = model._decode(z)
    fig, ax = plt.subplots(ncols=4)
    ax[0].set_title("Original View 1")
    ax[1].set_title("Original View 2")
    ax[2].set_title("Reconstruction View 1")
    ax[3].set_title("Reconstruction View 2")
    ax[0].imshow(x[0].detach().numpy().reshape((28, 28)))
    ax[1].imshow(y[0].detach().numpy().reshape((28, 28)))
    ax[2].imshow(recons[0].detach().numpy().reshape((28, 28)))
    ax[3].imshow(recons[1].detach().numpy().reshape((28, 28)))


n_train = 1024
n_val = 64
batch_size = 64
train_dataset = NoisyMNIST(
    root="", mnist_type="MNIST", train=True, flatten=True, download=True
)
val_dataset = Subset(train_dataset, np.arange(n_train, n_train + n_val))
train_dataset = Subset(train_dataset, np.arange(n_train))
train_loader, val_loader = get_dataloaders(
    train_dataset,
    val_dataset,
    batch_size=batch_size,
    val_batch_size=batch_size,
    shuffle_train=True,
)

# The number of latent dimensions across models
latent_dims = 20
# number of epochs for deep models
epochs = 10
# learning rate
lr = 0.0001
dropout = 0.1
layer_sizes = (1024, 1024, 1024)
"""
DVCCA
"""
encoder_1 = architectures.Encoder(
    latent_dims=latent_dims,
    feature_size=784,
    variational=True,
    layer_sizes=layer_sizes,
    dropout=dropout,
)
decoder_1 = architectures.Decoder(
    latent_dims=latent_dims, feature_size=784, layer_sizes=layer_sizes, dropout=dropout
)
decoder_2 = architectures.Decoder(
    latent_dims=latent_dims, feature_size=784, layer_sizes=layer_sizes, dropout=dropout
)
dvcca = DVCCA(
    latent_dims=latent_dims,
    encoders=[encoder_1],
    decoders=[decoder_1, decoder_2],
    lr=lr,
)
trainer = pl.Trainer(
    max_epochs=epochs,
    enable_checkpointing=False,
    log_every_n_steps=1,
    flush_logs_every_n_steps=1,
)
trainer.fit(dvcca, train_loader, val_loader)
dvcca.plot_latent_label(train_loader)
plot_reconstruction(dvcca, train_loader)
plt.suptitle("DVCCA")
plt.show()

"""
DVCCA Private
"""
private_encoder_1 = architectures.Encoder(
    latent_dims=latent_dims,
    feature_size=784,
    variational=True,
    layer_sizes=layer_sizes,
    dropout=dropout,
)
private_encoder_2 = architectures.Encoder(
    latent_dims=latent_dims,
    feature_size=784,
    variational=True,
    layer_sizes=layer_sizes,
    dropout=dropout,
)
private_decoder_1 = architectures.Decoder(
    latent_dims=2 * latent_dims,
    feature_size=784,
    layer_sizes=layer_sizes,
    dropout=dropout,
)
private_decoder_2 = architectures.Decoder(
    latent_dims=2 * latent_dims,
    feature_size=784,
    layer_sizes=layer_sizes,
    dropout=dropout,
)
dvccap = DVCCA(
    latent_dims=latent_dims,
    encoders=[encoder_1],
    decoders=[private_decoder_1, private_decoder_2],
    private_encoders=[private_encoder_1, private_encoder_2],
    lr=lr,
)
trainer = pl.Trainer(
    max_epochs=epochs,
    enable_checkpointing=False,
    log_every_n_steps=1,
    flush_logs_every_n_steps=1,
)
trainer.fit(dvccap, train_loader, val_loader)
plot_reconstruction(dvccap, train_loader)
plt.suptitle("DVCCA Private")
plt.show()
"""
DCCAE
"""
encoder_1 = architectures.Encoder(
    latent_dims=latent_dims, feature_size=784, layer_sizes=layer_sizes
)
encoder_2 = architectures.Encoder(
    latent_dims=latent_dims, feature_size=784, layer_sizes=layer_sizes
)
decoder_1 = architectures.Decoder(
    latent_dims=latent_dims, feature_size=784, layer_sizes=layer_sizes, dropout=dropout
)
decoder_2 = architectures.Decoder(
    latent_dims=latent_dims, feature_size=784, layer_sizes=layer_sizes, dropout=dropout
)
dccae = DCCAE(
    latent_dims=latent_dims,
    encoders=[encoder_1, encoder_2],
    decoders=[decoder_1, decoder_2],
    lr=lr,
    lam=0.5,
    optimizer="adam",
)
trainer = pl.Trainer(
    max_epochs=epochs,
    enable_checkpointing=False,
    log_every_n_steps=1,
    flush_logs_every_n_steps=1,
)
trainer.fit(dccae, train_loader, val_loader)
dccae.plot_latent_label(train_loader)
plt.suptitle("DCCAE")
plot_reconstruction(dccae, train_loader)
plt.suptitle("DCCAE")
plt.show()

"""
SplitAE
"""
encoder_1 = architectures.Encoder(
    latent_dims=latent_dims, feature_size=784, layer_sizes=layer_sizes
)
decoder_1 = architectures.Decoder(
    latent_dims=latent_dims, feature_size=784, layer_sizes=layer_sizes, dropout=dropout
)
decoder_2 = architectures.Decoder(
    latent_dims=latent_dims, feature_size=784, layer_sizes=layer_sizes, dropout=dropout
)
splitae = SplitAE(
    latent_dims=latent_dims,
    encoder=encoder_1,
    decoders=[decoder_1, decoder_2],
    lr=lr,
    optimizer="adam",
)
trainer = pl.Trainer(
    max_epochs=epochs,
    enable_checkpointing=False,
    log_every_n_steps=1,
    flush_logs_every_n_steps=1,
)
trainer.fit(splitae, train_loader, val_loader)
plt.suptitle("SplitAE")
plot_reconstruction(splitae, train_loader)
plt.suptitle("SplitAE")
plt.show()
