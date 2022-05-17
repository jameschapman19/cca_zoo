"""
Deep CCA with more customisation
==================================

Showing some examples of more advanced functionality with DCCA and pytorch-lightning
"""

# %%
import pytorch_lightning as pl
from matplotlib import pyplot as plt

from cca_zoo.deepmodels import DCCA, _architectures
from cca_zoo.utils import pairplot_label
from examples.utils import example_mnist_data

LATENT_DIMS = 2
EPOCHS = 10
N_TRAIN = 500
N_VAL = 100

train_loader, val_loader, train_labels = example_mnist_data(N_TRAIN, N_VAL)

# TODO add in custom architecture and schedulers and stuff to show it off
encoder_1 = _architectures.Encoder(latent_dims=LATENT_DIMS, feature_size=392)
encoder_2 = _architectures.Encoder(latent_dims=LATENT_DIMS, feature_size=392)

# Deep CCA
dcca = DCCA(
    latent_dims=LATENT_DIMS, encoders=[encoder_1, encoder_2], scheduler="cosine"
)
trainer = pl.Trainer(max_epochs=EPOCHS, enable_checkpointing=False)
trainer.fit(dcca, train_loader, val_loader)
pairplot_label(dcca.transform(train_loader), train_labels)
plt.suptitle("DCCA by Barlow Twins")
plt.show()
