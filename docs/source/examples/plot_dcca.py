"""
Deep CCA
===========================

This example demonstrates how to easily train Deep CCA linear and variants
using cca_zoo, a library for canonical correlation analysis and related methods.
"""

# %%
# Imports
# -------

import pytorch_lightning as pl
from matplotlib import pyplot as plt
from cca_zoo.deep import DCCA, DCCA_EY, DCCA_NOI, DCCA_SDL, BarlowTwins, architectures
from cca_zoo.visualisation import Plotter
from docs.source.examples import example_mnist_data

# %%
# Data
# -----
# We use the MNIST dataset as an example of two views of the same data.
# We split the images into two halves and treat them as separate views.

LATENT_DIMS = 2  # The dimensionality of the latent space
EPOCHS = 10  # The number of epochs to train the models
N_TRAIN = 500  # The number of training samples
N_VAL = 100  # The number of validation samples

train_loader, val_loader, train_labels, val_labels = example_mnist_data(N_TRAIN, N_VAL)

encoder_1 = architectures.Encoder(latent_dimensions=LATENT_DIMS, feature_size=392)
encoder_2 = architectures.Encoder(latent_dimensions=LATENT_DIMS, feature_size=392)

# %%
# Deep CCA
# ----------------------------
# Deep CCA is a method that learns nonlinear transformations of two views
# such that the resulting latent representations are maximally correlated.

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
# Deep CCA EY is a variant of Deep CCA that uses an explicit objective function
# based on the eigenvalue decomposition of the cross-covariance matrix.

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
plt.show()

# %%
# Deep CCA by Non-Linear Orthogonal Iterations
# ----------------------------------------------
# Deep CCA by Non-Linear Orthogonal Iterations (DCCA_NOI) is another variant of Deep CCA
# that uses an iterative algorithm to orthogonalize the latent representations.

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
# Deep CCA by Stochastic Decorrelation Loss (DCCA_SDL) is yet another variant of Deep CCA
# that uses a stochastic gradient descent algorithm to minimize a decorrelation loss function.

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
plt.show()

# %%
# Deep CCA by Barlow Twins
# ----------------------------------------------
# Deep CCA by Barlow Twins is a self-supervised learning method that learns representations
# that are invariant to augmentations of the same data. It can be seen as a special case of Deep CCA
# where the two views are random augmentations of the same input.

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
