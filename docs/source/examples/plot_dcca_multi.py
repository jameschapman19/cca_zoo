"""
Multiview Deep CCA Extensions
=============================

This script showcases how to train extensions of Deep Canonical Correlation Analysis
(Deep CCA) that can handle more than two views of data, using CCA-Zoo's functionalities.

Features:
- Deep MCCA (Multiset CCA)
- Deep GCCA (Generalized CCA)
- Deep TCCA (Tied CCA)

"""

import pytorch_lightning as pl
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
# Deep MCCA (Multiset CCA)
# ------------------------
# A multiview extension of CCA, aiming to find latent spaces that are maximally correlated across multiple views.

dcca_mcca = DCCA(
    latent_dimensions=LATENT_DIMS,
    encoders=[encoder_1, encoder_2],
    objective=objectives.MCCA,
)
trainer_mcca = pl.Trainer(max_epochs=EPOCHS, enable_checkpointing=False)
trainer_mcca.fit(dcca_mcca, train_loader, val_loader)

# %%
# Deep GCCA (Generalized CCA)
# ---------------------------
# A method that finds projections of multiple views such that the variance explained
# by the canonical components is maximized.

dcca_gcca = DCCA(
    latent_dimensions=LATENT_DIMS,
    encoders=[encoder_1, encoder_2],
    objective=objectives.GCCA,
)
trainer_gcca = pl.Trainer(max_epochs=EPOCHS, enable_checkpointing=False)
trainer_gcca.fit(dcca_gcca, train_loader, val_loader)

# %%
# Deep TCCA (Tied CCA)
# --------------------
# An approach where views share the same weight parameters during training.

dcca_tcca = DTCCA(latent_dimensions=LATENT_DIMS, encoders=[encoder_1, encoder_2])
trainer_tcca = pl.Trainer(max_epochs=EPOCHS, enable_checkpointing=False)
trainer_tcca.fit(dcca_tcca, train_loader, val_loader)
