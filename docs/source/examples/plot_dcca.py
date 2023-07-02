"""
Deep CCA
===========================

This example demonstrates how to easily train Deep CCA linear and variants
"""

import pytorch_lightning as pl
from matplotlib import pyplot as plt

# %%
from cca_zoo.deep import (
    DCCA,
    DCCA_EY,
    DCCA_NOI,
    DCCA_SDL,
    BarlowTwins,
    architectures,
)
from cca_zoo.visualisation import Plotter

from docs.source.examples import example_mnist_data

# %%
# Data
# -----
LATENT_DIMS = 2
EPOCHS = 10
N_TRAIN = 500
N_VAL = 100

train_loader, val_loader, train_labels, val_labels = example_mnist_data(N_TRAIN, N_VAL)

encoder_1 = architectures.Encoder(latent_dimensions=LATENT_DIMS, feature_size=392)
encoder_2 = architectures.Encoder(latent_dimensions=LATENT_DIMS, feature_size=392)

# %%
# Deep CCA
# ----------------------------
dcca = DCCA(latent_dimensions=LATENT_DIMS, encoders=[encoder_1, encoder_2])
trainer = pl.Trainer(
    max_epochs=EPOCHS,
    enable_checkpointing=False,
    log_every_n_steps=1,
)
trainer.fit(dcca, train_loader, val_loader)
plotter = Plotter()
ax = plotter.plot_scores_multi(dcca.transform(val_loader), labels=val_labels)
ax.fig.suptitle("Deep CCA")
plt.show()

# %%
# Deep CCA EY
# ----------------------------
dcca_eg = DCCA_EY(
    latent_dimensions=LATENT_DIMS, encoders=[encoder_1, encoder_2], lr=1e-3
)
trainer = pl.Trainer(
    max_epochs=EPOCHS,
    enable_checkpointing=False,
    log_every_n_steps=1,
)
trainer.fit(dcca_eg, train_loader, val_loader)
plotter = Plotter()
ax = plotter.plot_scores_multi(dcca_eg.transform(val_loader), labels=val_labels)
ax.fig.suptitle("Deep CCA EY")

# %%
# Deep CCA by Non-Linear Orthogonal Iterations
# ----------------------------------------------
dcca_noi = DCCA_NOI(
    latent_dimensions=LATENT_DIMS, N=N_TRAIN, encoders=[encoder_1, encoder_2]
)
trainer = pl.Trainer(
    max_epochs=EPOCHS,
    enable_checkpointing=False,
    log_every_n_steps=1,
)
trainer.fit(dcca_noi, train_loader, val_loader)
plotter = Plotter()
ax = plotter.plot_scores_multi(dcca_noi.transform(val_loader), labels=val_labels)
ax.fig.suptitle("Deep CCA by Non-Linear Orthogonal Iterations")
plt.show()

# %%
# Deep CCA by Stochastic Decorrelation Loss
# ----------------------------------------------
dcca_sdl = DCCA_SDL(
    latent_dimensions=LATENT_DIMS, N=N_TRAIN, encoders=[encoder_1, encoder_2]
)
trainer = pl.Trainer(
    max_epochs=EPOCHS,
    enable_checkpointing=False,
    log_every_n_steps=1,
)
trainer.fit(dcca_sdl, train_loader, val_loader)
plotter = Plotter()
ax = plotter.plot_scores_multi(dcca_sdl.transform(val_loader), labels=val_labels)
ax.fig.suptitle("Deep CCA by Stochastic Decorrelation Loss")

# %%
# Deep CCA by Barlow Twins
# ----------------------------------------------
barlowtwins = BarlowTwins(
    latent_dimensions=LATENT_DIMS, encoders=[encoder_1, encoder_2]
)
trainer = pl.Trainer(
    max_epochs=EPOCHS,
    enable_checkpointing=False,
    log_every_n_steps=1,
)
trainer.fit(barlowtwins, train_loader, val_loader)
plotter = Plotter()
ax = plotter.plot_scores_multi(barlowtwins.transform(val_loader), labels=val_labels)
ax.fig.suptitle("Deep CCA by Barlow Twins")
plt.show()
