import numpy as np
import pytorch_lightning as pl
from sklearn.utils.validation import check_random_state
from torch import manual_seed
from torch.utils.data import random_split

from cca_zoo import data
from cca_zoo.deepmodels import (
    DCCA,
    DCCAE,
    DVCCA,
    DCCA_NOI,
    DTCCA,
    SplitAE,
    get_dataloaders,
    BarlowTwins,
    DCCA_SDL,
)
from cca_zoo.deepmodels import objectives, architectures
from cca_zoo.models import CCA

manual_seed(0)
rng = check_random_state(0)
X = rng.rand(256, 10)
Y = rng.rand(256, 12)
Z = rng.rand(256, 14)
X_conv = rng.rand(256, 1, 16, 16)
Y_conv = rng.rand(256, 1, 16, 16)
dataset = data.CCA_Dataset([X, Y, Z])
train_dataset, val_dataset = random_split(dataset, [200, 56])
loader = get_dataloaders(dataset)
train_loader, val_loader = get_dataloaders(train_dataset, val_dataset)
conv_dataset = data.CCA_Dataset((X_conv, Y_conv))
conv_loader = get_dataloaders(conv_dataset)
train_ids = train_dataset.indices


def test_DCCA_methods():
    N = len(train_dataset)
    latent_dims = 2
    epochs = 100
    cca = CCA(latent_dims=latent_dims).fit((X, Y))
    # DCCA
    encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
    encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=12)
    dcca = DCCA(
        latent_dims=latent_dims,
        encoders=[encoder_1, encoder_2],
        objective=objectives.CCA,
    )
    trainer = pl.Trainer(
        max_epochs=epochs, log_every_n_steps=1, enable_checkpointing=False
    )
    trainer.fit(dcca, train_loader, val_dataloaders=val_loader)
    assert (
            np.testing.assert_array_less(
                cca.score((X, Y)).sum(), dcca.score(train_loader, train=True).sum()
            )
            is None
    )
    # DCCA_NOI
    encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
    encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=12)
    dcca_noi = DCCA_NOI(latent_dims, N, encoders=[encoder_1, encoder_2], rho=0)
    trainer = pl.Trainer(
        max_epochs=epochs, log_every_n_steps=1, enable_checkpointing=False
    )
    trainer.fit(dcca_noi, train_loader)
    assert (
        np.testing.assert_array_less(
            cca.score((X, Y)).sum(), dcca_noi.score(train_loader).sum()
        )
        is None
    )
    # Soft Decorrelation (_stochastic Decorrelation Loss)
    encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
    encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=12)
    sdl = DCCA_SDL(latent_dims, N, encoders=[encoder_1, encoder_2], lam=1e-2, lr=1e-3)
    trainer = pl.Trainer(max_epochs=epochs, log_every_n_steps=1)
    trainer.fit(sdl, train_loader)
    assert (
        np.testing.assert_array_less(
            cca.score((X, Y)).sum(), sdl.score(train_loader).sum()
        )
        is None
    )
    # Barlow Twins
    encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
    encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=12)
    barlowtwins = BarlowTwins(
        latent_dims=latent_dims,
        encoders=[encoder_1, encoder_2],
    )
    trainer = pl.Trainer(
        max_epochs=epochs, log_every_n_steps=1, enable_checkpointing=False
    )
    trainer.fit(barlowtwins, train_loader)
    assert (
        np.testing.assert_array_less(
            cca.score((X, Y)).sum(), barlowtwins.score(train_loader).sum()
        )
        is None
    )
    # DGCCA
    encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
    encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=12)
    dgcca = DCCA(
        latent_dims=latent_dims,
        encoders=[encoder_1, encoder_2],
        objective=objectives.GCCA,
    )
    trainer = pl.Trainer(
        max_epochs=epochs, log_every_n_steps=1, enable_checkpointing=False
    )
    trainer.fit(dgcca, train_loader)
    assert (
        np.testing.assert_array_less(
            cca.score((X, Y)).sum(), dgcca.score(train_loader).sum()
        )
        is None
    )
    # DMCCA
    encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
    encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=12)
    dmcca = DCCA(
        latent_dims=latent_dims,
        encoders=[encoder_1, encoder_2],
        objective=objectives.MCCA,
    )
    trainer = pl.Trainer(
        max_epochs=epochs, log_every_n_steps=1, enable_checkpointing=False
    )
    trainer.fit(dmcca, train_loader)
    assert (
        np.testing.assert_array_less(
            cca.score((X, Y)).sum(), dmcca.score(train_loader).sum()
        )
        is None
    )


def test_DTCCA_methods():
    # check that DTCCA is equivalent to CCA for 2 views with linear encoders
    latent_dims = 2
    epochs = 150
    cca = CCA(latent_dims=latent_dims)
    encoder_1 = architectures.LinearEncoder(latent_dims=latent_dims, feature_size=10)
    encoder_2 = architectures.LinearEncoder(latent_dims=latent_dims, feature_size=12)
    dtcca = DTCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2], lr=1e-2)
    trainer = pl.Trainer(max_epochs=epochs, enable_checkpointing=False)
    trainer.fit(dtcca, train_loader)
    z = dtcca.transform(train_loader)
    assert (
        np.testing.assert_array_almost_equal(
            cca.fit((X[train_ids], Y[train_ids]))
            .score((X[train_ids], Y[train_ids]))
            .sum(),
            cca.fit((z)).score((z)).sum(),
            decimal=1,
        )
        is None
    )


def test_DCCAE_methods():
    latent_dims = 2
    encoder_1 = architectures.CNNEncoder(latent_dims=latent_dims, feature_size=(16, 16))
    encoder_2 = architectures.CNNEncoder(latent_dims=latent_dims, feature_size=(16, 16))
    decoder_1 = architectures.CNNDecoder(latent_dims=latent_dims, feature_size=(16, 16))
    decoder_2 = architectures.CNNDecoder(latent_dims=latent_dims, feature_size=(16, 16))
    # SplitAE
    splitae = SplitAE(
        latent_dims=latent_dims, encoder=encoder_1, decoders=[decoder_1, decoder_2]
    )
    trainer = pl.Trainer(max_epochs=5, enable_checkpointing=False)
    trainer.fit(splitae, conv_loader)
    # DCCAE
    dccae = DCCAE(
        latent_dims=latent_dims,
        encoders=[encoder_1, encoder_2],
        decoders=[decoder_1, decoder_2],
    )
    trainer = pl.Trainer(max_epochs=5, enable_checkpointing=False)
    trainer.fit(dccae, conv_loader)


def test_DVCCA_p_methods():
    latent_dims = 2
    encoder_1 = architectures.Encoder(
        latent_dims=latent_dims, feature_size=10, variational=True
    )
    encoder_2 = architectures.Encoder(
        latent_dims=latent_dims, feature_size=12, variational=True
    )
    private_encoder_1 = architectures.Encoder(
        latent_dims=latent_dims, feature_size=10, variational=True
    )
    private_encoder_2 = architectures.Encoder(
        latent_dims=latent_dims, feature_size=12, variational=True
    )
    decoder_1 = architectures.Decoder(latent_dims=2 * latent_dims, feature_size=10)
    decoder_2 = architectures.Decoder(latent_dims=2 * latent_dims, feature_size=12)
    # DVCCA
    dvcca = DVCCA(
        latent_dims=latent_dims,
        encoders=[encoder_1, encoder_2],
        decoders=[decoder_1, decoder_2],
        private_encoders=[private_encoder_1, private_encoder_2],
    )
    trainer = pl.Trainer(max_epochs=5, log_every_n_steps=1, enable_checkpointing=False)
    trainer.fit(dvcca, train_loader)
    dvcca.transform(train_loader)


def test_DVCCA_methods():
    latent_dims = 2
    encoder_1 = architectures.Encoder(
        latent_dims=latent_dims, feature_size=10, variational=True
    )
    encoder_2 = architectures.Encoder(
        latent_dims=latent_dims, feature_size=12, variational=True
    )
    decoder_1 = architectures.Decoder(latent_dims=latent_dims, feature_size=10)
    decoder_2 = architectures.Decoder(latent_dims=latent_dims, feature_size=12)
    dvcca = DVCCA(
        latent_dims=latent_dims,
        encoders=[encoder_1, encoder_2],
        decoders=[decoder_1, decoder_2],
    )
    trainer = pl.Trainer(max_epochs=5, log_every_n_steps=1, enable_checkpointing=False)
    trainer.fit(dvcca, train_loader)


def test_linear():
    encoder_1 = architectures.LinearEncoder(latent_dims=1, feature_size=10)
    encoder_2 = architectures.LinearEncoder(latent_dims=1, feature_size=12)
    dcca = DCCA(latent_dims=1, encoders=[encoder_1, encoder_2], lr=1e-1)
    trainer = pl.Trainer(max_epochs=100, enable_checkpointing=False)
    trainer.fit(dcca, loader)
    cca = CCA().fit((X, Y))
    # check linear encoder with SGD matches vanilla linear CCA
    assert (
        np.testing.assert_array_almost_equal(
            cca.score((X, Y)), dcca.score(loader), decimal=2
        )
        is None
    )
