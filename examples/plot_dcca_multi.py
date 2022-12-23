"""
Deep CCA for more than 2 views
=================================

This example demonstrates how to easily train Deep CCA models and variants
"""

import pytorch_lightning as pl

from cca_zoo.deepmodels import (
    DCCA,
    DTCCA,
    objectives,
)
from cca_zoo.deepmodels import architectures

# %%
# Data
# -----
from examples import example_mnist_data

LATENT_DIMS = 2
EPOCHS = 10
N_TRAIN = 500
N_VAL = 100

train_loader, val_loader, train_labels = example_mnist_data(N_TRAIN, N_VAL)

encoder_1 = architectures.Encoder(latent_dims=LATENT_DIMS, feature_size=392)
encoder_2 = architectures.Encoder(latent_dims=LATENT_DIMS, feature_size=392)

# %%
# Deep MCCA
# ----------
dcca = DCCA(
    latent_dims=LATENT_DIMS, encoders=[encoder_1, encoder_2], objective=objectives.MCCA
)
trainer = pl.Trainer(max_epochs=EPOCHS, enable_checkpointing=False)
trainer.fit(dcca, train_loader, val_loader)

# %%
# Deep GCCA
# ---------
dcca = DCCA(
    latent_dims=LATENT_DIMS, encoders=[encoder_1, encoder_2], objective=objectives.GCCA
)
trainer = pl.Trainer(max_epochs=EPOCHS, enable_checkpointing=False)
trainer.fit(dcca, train_loader, val_loader)

# %%
# Deep TCCA
# ---------
dcca = DTCCA(latent_dims=LATENT_DIMS, encoders=[encoder_1, encoder_2])
trainer = pl.Trainer(max_epochs=EPOCHS, enable_checkpointing=False)
trainer.fit(dcca, train_loader, val_loader)
