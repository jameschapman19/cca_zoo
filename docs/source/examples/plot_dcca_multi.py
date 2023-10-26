"""
Multiview Deep CCALoss Extensions
=============================

This script showcases how to train extensions of Deep Canonical Correlation Analysis
(Deep CCALoss) that can handle more than two representations of data, using CCALoss-Zoo's functionalities.

Features:
- Deep MCCALoss (Multiset CCALoss)
- Deep GCCALoss (Generalized CCALoss)
- Deep TCCALoss (Tied CCALoss)

"""

import lightning.pytorch as pl
from cca_zoo.deep import DCCA, DTCCA, architectures, objectives

# %%
# Data Preparation
# ----------------
# Here, we use a segmented MNIST dataset as an example of multiview data.

from docs.source.examples import example_mnist_data

LATENT_DIMS = 2
EPOCHS = 10
N_TRAIN = 500
N_VAL = 100

train_loader, val_loader, train_labels, val_labels = example_mnist_data(N_TRAIN, N_VAL)

encoder_1 = architectures.Encoder(latent_dimensions=LATENT_DIMS, feature_size=392)
encoder_2 = architectures.Encoder(latent_dimensions=LATENT_DIMS, feature_size=392)

# %%
# Deep MCCALoss (Multiset CCALoss)
# ------------------------
# A multiview extension of CCALoss, aiming to find latent spaces that are maximally correlated across multiple representations.

dcca_mcca = DCCA(
    latent_dimensions=LATENT_DIMS,
    encoders=[encoder_1, encoder_2],
    objective=objectives.MCCALoss,
)
trainer_mcca = pl.Trainer(max_epochs=EPOCHS, enable_checkpointing=False, enable_model_summary=False,enable_progress_bar=False)
trainer_mcca.fit(dcca_mcca, train_loader, val_loader)

# %%
# Deep GCCALoss (Generalized CCALoss)
# ---------------------------
# A method that finds projections of multiple representations such that the variance explained
# by the canonical components is maximized.

dcca_gcca = DCCA(
    latent_dimensions=LATENT_DIMS,
    encoders=[encoder_1, encoder_2],
    objective=objectives.GCCALoss,
)
trainer_gcca = pl.Trainer(max_epochs=EPOCHS, enable_checkpointing=False, enable_model_summary=False,enable_progress_bar=False)
trainer_gcca.fit(dcca_gcca, train_loader, val_loader)

# %%
# Deep TCCALoss (Tied CCALoss)
# --------------------
# An approach where representations share the same weight parameters during training.

dcca_tcca = DTCCA(latent_dimensions=LATENT_DIMS, encoders=[encoder_1, encoder_2])
trainer_tcca = pl.Trainer(max_epochs=EPOCHS, enable_checkpointing=False, enable_model_summary=False,enable_progress_bar=False)
trainer_tcca.fit(dcca_tcca, train_loader, val_loader)
