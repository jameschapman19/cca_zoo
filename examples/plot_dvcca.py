"""
Deep Variational CCA and Deep Canonically Correlated Autoencoders
====================================================================

This example demonstrates multiview models which can reconstruct their inputs
"""
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from cca_zoo.deepmodels import DCCAE, DVCCA, SplitAE
from cca_zoo.deepmodels import architectures
from cca_zoo.plotting import tsne_label
from examples import example_mnist_data


def plot_reconstruction(model, loader):
    recons = model.recon(loader, mle=True)
    originals = loader.dataset.dataset[0]["views"]
    n_cols = 2
    fig, ax = plt.subplots(ncols=n_cols, nrows=2)
    for i, (original) in enumerate(originals):
        ax[i, 0].set_title(f"Original View {i}")
        ax[i, 1].set_title(f"Mean View {i}")
        ax[i, 0].imshow(original.reshape((28, 28)))
        ax[i, 1].imshow(recons[i][0].reshape((28, 28)))


# %%
# Data
# -----
LATENT_DIMS = 2
EPOCHS = 1
N_TRAIN = 500
N_VAL = 100
lr = 0.0001
dropout = 0.1
layer_sizes = (1024, 1024, 1024)

train_loader, val_loader, train_labels = example_mnist_data(
    N_TRAIN, N_VAL, type="noisy"
)

# %%
# Deep Variational CCA
# ----------------------------
encoder_1 = architectures.Encoder(
    latent_dims=LATENT_DIMS,
    feature_size=784,
    variational=True,
    layer_sizes=layer_sizes,
    dropout=dropout,
)
decoder_1 = architectures.Decoder(
    latent_dims=LATENT_DIMS, feature_size=784, layer_sizes=layer_sizes, dropout=dropout
)
decoder_2 = architectures.Decoder(
    latent_dims=LATENT_DIMS, feature_size=784, layer_sizes=layer_sizes, dropout=dropout
)
dvcca = DVCCA(
    latent_dims=LATENT_DIMS,
    encoders=[encoder_1],
    decoders=[decoder_1, decoder_2],
    lr=lr,
)
trainer = pl.Trainer(
    max_epochs=EPOCHS,
    enable_checkpointing=False,
    log_every_n_steps=1,
)
trainer.fit(dvcca, train_loader, val_loader)
tsne_label(dvcca.transform(train_loader)["shared"], train_labels)
plot_reconstruction(dvcca, train_loader)
plt.suptitle("DVCCA")
plt.show()

# %%
# Deep Variational CCA (Private)
# -------------------------------
private_encoder_1 = architectures.Encoder(
    latent_dims=LATENT_DIMS,
    feature_size=784,
    variational=True,
    layer_sizes=layer_sizes,
    dropout=dropout,
)
private_encoder_2 = architectures.Encoder(
    latent_dims=LATENT_DIMS,
    feature_size=784,
    variational=True,
    layer_sizes=layer_sizes,
    dropout=dropout,
)
private_decoder_1 = architectures.Decoder(
    latent_dims=2 * LATENT_DIMS,
    feature_size=784,
    layer_sizes=layer_sizes,
    dropout=dropout,
)
private_decoder_2 = architectures.Decoder(
    latent_dims=2 * LATENT_DIMS,
    feature_size=784,
    layer_sizes=layer_sizes,
    dropout=dropout,
)
dvccap = DVCCA(
    latent_dims=LATENT_DIMS,
    encoders=[encoder_1],
    decoders=[private_decoder_1, private_decoder_2],
    private_encoders=[private_encoder_1, private_encoder_2],
    lr=lr,
)
trainer = pl.Trainer(
    max_epochs=EPOCHS,
    enable_checkpointing=False,
    log_every_n_steps=1,
)
trainer.fit(dvccap, train_loader, val_loader)
tsne_label(dvccap.transform(train_loader)["shared"], train_labels)
plot_reconstruction(dvccap, train_loader)
plt.suptitle("DVCCA Private")
plt.show()

# %%
# Deep Canonically Correlated Autoencoders
# -----------------------------------------
encoder_1 = architectures.Encoder(
    latent_dims=LATENT_DIMS, feature_size=784, layer_sizes=layer_sizes
)
encoder_2 = architectures.Encoder(
    latent_dims=LATENT_DIMS, feature_size=784, layer_sizes=layer_sizes
)
decoder_1 = architectures.Decoder(
    latent_dims=LATENT_DIMS, feature_size=784, layer_sizes=layer_sizes, dropout=dropout
)
decoder_2 = architectures.Decoder(
    latent_dims=LATENT_DIMS, feature_size=784, layer_sizes=layer_sizes, dropout=dropout
)
dccae = DCCAE(
    latent_dims=LATENT_DIMS,
    encoders=[encoder_1, encoder_2],
    decoders=[decoder_1, decoder_2],
    lr=lr,
    lam=0.5,
    optimizer="adam",
)
trainer = pl.Trainer(
    max_epochs=EPOCHS,
    enable_checkpointing=False,
    log_every_n_steps=1,
)
trainer.fit(dccae, train_loader, val_loader)
tsne_label(dccae.transform(train_loader)[0], train_labels)
plot_reconstruction(dccae, train_loader)
plt.suptitle("DCCAE")
plt.show()

# %%
# Split Autoencoders
# -------------------
encoder_1 = architectures.Encoder(
    latent_dims=LATENT_DIMS, feature_size=784, layer_sizes=layer_sizes
)
decoder_1 = architectures.Decoder(
    latent_dims=LATENT_DIMS, feature_size=784, layer_sizes=layer_sizes, dropout=dropout
)
decoder_2 = architectures.Decoder(
    latent_dims=LATENT_DIMS, feature_size=784, layer_sizes=layer_sizes, dropout=dropout
)
splitae = SplitAE(
    latent_dims=LATENT_DIMS,
    encoder=encoder_1,
    decoders=[decoder_1, decoder_2],
    lr=lr,
    optimizer="adam",
)
trainer = pl.Trainer(
    max_epochs=EPOCHS,
    enable_checkpointing=False,
    log_every_n_steps=1,
)
trainer.fit(splitae, train_loader, val_loader)
tsne_label(splitae.transform(train_loader)[0], train_labels)
plot_reconstruction(splitae, train_loader)
plt.suptitle("SplitAE")
plt.show()
