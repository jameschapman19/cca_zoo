"""
Deep CCA
===========================

This example demonstrates how to easily train Deep CCA models and variants
"""

import pytorch_lightning as pl
from matplotlib import pyplot as plt

# %%
from cca_zoo.deepmodels import (
    DCCA,
    DCCA_NOI,
    DCCA_SDL,
    BarlowTwins,
    DCCA_EigenGame,
)
from cca_zoo.deepmodels import architectures
from cca_zoo.plotting import pairplot_label
from examples import example_mnist_data

# %%
# Data
# -----
LATENT_DIMS = 2
EPOCHS = 10
N_TRAIN = 500
N_VAL = 100

train_loader, val_loader, train_labels = example_mnist_data(N_TRAIN, N_VAL)

encoder_1 = architectures.Encoder(latent_dims=LATENT_DIMS, feature_size=392)
encoder_2 = architectures.Encoder(latent_dims=LATENT_DIMS, feature_size=392)

# %%
# Deep CCA
# ----------------------------
dcca = DCCA(latent_dims=LATENT_DIMS, encoders=[encoder_1, encoder_2])
trainer = pl.Trainer(
    max_epochs=EPOCHS,
    enable_checkpointing=False,
    log_every_n_steps=1,
)
trainer.fit(dcca, train_loader, val_loader)
pairplot_label(dcca.transform(train_loader), train_labels, title="DCCA")
plt.show()

# %%
# Deep CCA EigenGame
# ----------------------------
dcca_eg = DCCA_EigenGame(
    latent_dims=LATENT_DIMS, encoders=[encoder_1, encoder_2], lr=1e-5
)
trainer = pl.Trainer(
    max_epochs=EPOCHS,
    enable_checkpointing=False,
    log_every_n_steps=1,
)
trainer.fit(dcca_eg, train_loader, val_loader)
pairplot_label(dcca_eg.transform(train_loader), train_labels, title="DCCA-EigenGame")
plt.show()

# %%
# Deep CCA by Non-Linear Orthogonal Iterations
# ----------------------------------------------
dcca_noi = DCCA_NOI(latent_dims=LATENT_DIMS, N=N_TRAIN, encoders=[encoder_1, encoder_2])
trainer = pl.Trainer(
    max_epochs=EPOCHS,
    enable_checkpointing=False,
    log_every_n_steps=1,
)
trainer.fit(dcca_noi, train_loader, val_loader)
pairplot_label(
    dcca_noi.transform(train_loader),
    train_labels,
    title="DCCA by Non-Linear Orthogonal Iterations",
)
plt.show()

# %%
# Deep CCA by Stochastic Decorrelation Loss
# ----------------------------------------------
dcca_sdl = DCCA_SDL(latent_dims=LATENT_DIMS, N=N_TRAIN, encoders=[encoder_1, encoder_2])
trainer = pl.Trainer(
    max_epochs=EPOCHS,
    enable_checkpointing=False,
    log_every_n_steps=1,
)
trainer.fit(dcca_sdl, train_loader, val_loader)
pairplot_label(
    dcca_sdl.transform(train_loader),
    train_labels,
    title="DCCA by Stochastic Decorrelation",
)
plt.show()

# %%
# Deep CCA by Barlow Twins
# ----------------------------------------------
barlowtwins = BarlowTwins(latent_dims=LATENT_DIMS, encoders=[encoder_1, encoder_2])
trainer = pl.Trainer(
    max_epochs=EPOCHS,
    enable_checkpointing=False,
    log_every_n_steps=1,
)
trainer.fit(barlowtwins, train_loader, val_loader)
pairplot_label(
    barlowtwins.transform(train_loader), train_labels, title="DCCA by Barlow Twins"
)
plt.show()
