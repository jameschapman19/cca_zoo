"""
Deep Canonical Correlation Analysis (CCA) using `cca_zoo`
========================================================

This script showcases how to implement various Deep CCA methods and their
variants using the `cca_zoo` library, a dedicated tool for canonical
correlation analysis and its related techniques. The MNIST dataset is used
as an example, where images are split into two halves to treat as separate views.

Key Features:
- Demonstrates the training process of multiple Deep CCA variants.
- Visualizes the results of each variant for comparative analysis.
- Leverages `cca_zoo` for canonical correlation analysis techniques.
"""


# %%
# Imports
# -------

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from matplotlib import pyplot as plt
from cca_zoo.deep import DCCA, DCCA_EY, DCCA_NOI, DCCA_SDL, BarlowTwins, architectures
from cca_zoo.deep._discriminative import VICReg
from cca_zoo.visualisation import (
    ScoreScatterDisplay,
    UMAPScoreDisplay,
    TSNEScoreDisplay,
)
from docs.source.examples import example_mnist_data

# %%
# Data
# -----
# We use the MNIST dataset as an example of two views of the same data.
# We split the images into two halves and treat them as separate views.

seed_everything(42)
LATENT_DIMS = 2  # The dimensionality of the latent space
EPOCHS = 10  # The number of epochs to train the models
N_TRAIN = 1000  # The number of training samples
N_VAL = 200  # The number of validation samples
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
)
trainer.fit(dcca, train_loader, val_loader)
# Visualizing the Latent Space
# ----------------------------
# After training, we can visualize the learned representations in the latent space.
# For this, we provide three options: Scatter, UMAP and t-SNE.

# Scatterplot of the latent space
score_display = ScoreScatterDisplay.from_estimator(
    dcca, val_loader, labels=val_labels.astype(str)
)
score_display.plot()
plt.show()

# UMAP Visualization
score_display = UMAPScoreDisplay.from_estimator(
    dcca, val_loader, labels=val_labels.astype(str)
)
score_display.plot()
score_display.figure_.suptitle("UMAP Deep CCA")
plt.show()

# t-SNE Visualization
score_display = TSNEScoreDisplay.from_estimator(
    dcca, val_loader, labels=val_labels.astype(str)
)
score_display.plot()
score_display.figure_.suptitle("TSNE Deep CCA")
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
)
trainer.fit(dcca_eg, train_loader, val_loader)
score_display = ScoreScatterDisplay.from_estimator(
    dcca_eg, val_loader, labels=val_labels.astype(str)
)
score_display.plot()
plt.show()

# %%
# Deep CCA by Non-Linear Orthogonal Iterations
# ----------------------------------------------
# Deep CCA by Non-Linear Orthogonal Iterations (DCCA_NOI) is another variant of Deep CCA
# that uses an iterative algorithm to orthogonalize the latent representations.

dcca_noi = DCCA_NOI(latent_dimensions=LATENT_DIMS, encoders=[encoder_1, encoder_2])
trainer = pl.Trainer(
    max_epochs=EPOCHS,
    enable_checkpointing=False,
)
trainer.fit(dcca_noi, train_loader, val_loader)
score_display = ScoreScatterDisplay.from_estimator(
    dcca_noi, val_loader, labels=val_labels.astype(str)
)
score_display.plot()
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
)
trainer.fit(dcca_sdl, train_loader, val_loader)
score_display = ScoreScatterDisplay.from_estimator(
    dcca_sdl, val_loader, labels=val_labels.astype(str)
)
score_display.plot()
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
)
trainer.fit(barlowtwins, train_loader, val_loader)
score_display = ScoreScatterDisplay.from_estimator(
    barlowtwins, val_loader, labels=val_labels.astype(str)
)
score_display.plot()
plt.show()

# %%
# Deep CCA by VICReg
# ----------------------------------------------
# Deep CCA by VICReg is a self-supervised learning method that learns representations
# that are invariant to distortions of the same data. It can be seen as a special case of Deep CCA
# where the two views are random distortions of the same input.

dcca_vicreg = DCCA_SDL(
    latent_dimensions=LATENT_DIMS, N=N_TRAIN, encoders=[encoder_1, encoder_2]
)
trainer = pl.Trainer(
    max_epochs=EPOCHS,
    enable_checkpointing=False,
)
trainer.fit(dcca_vicreg, train_loader, val_loader)
score_display = ScoreScatterDisplay.from_estimator(
    dcca_vicreg, val_loader, labels=val_labels.astype(str)
)
score_display.plot()
plt.show()
