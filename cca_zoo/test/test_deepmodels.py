import numpy as np
import pytorch_lightning as pl
from sklearn.utils.validation import check_random_state
from torch import optim, manual_seed

from cca_zoo import data
from cca_zoo.deepmodels import (
    DCCA,
    DCCAE,
    DVCCA,
    DCCA_NOI,
    DTCCA,
    SplitAE,
    CCALightning,
    get_dataloaders,
    process_data,
    BarlowTwins,
    DCCA_SDL,
)
from cca_zoo.deepmodels import objectives, architectures
from cca_zoo.models import CCA

manual_seed(0)
rng = check_random_state(0)
X = rng.rand(200, 10)
Y = rng.rand(200, 12)
Z = rng.rand(200, 14)
X_conv = rng.rand(200, 1, 16, 16)
Y_conv = rng.rand(200, 1, 16, 16)
dataset = data.CCA_Dataset([X, Y, Z])
train_dataset, val_dataset = process_data(dataset, val_split=0.2)
train_dataset_numpy, val_dataset_numpy = process_data((X, Y, Z), val_split=0.2)
loader = get_dataloaders(dataset)
train_loader, val_loader = get_dataloaders(train_dataset, val_dataset)
train_loader_numpy, val_loader_numpy = get_dataloaders(train_dataset, val_dataset)
conv_dataset = data.CCA_Dataset((X_conv, Y_conv))
conv_loader = get_dataloaders(conv_dataset)


def test_DCCA_methods():
    N = len(train_dataset)
    latent_dims = 2
    epochs = 100
    cca = CCA(latent_dims=latent_dims).fit((X, Y))
    # DCCA_NOI
    encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
    encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=12)
    dcca_noi = DCCA_NOI(latent_dims, N, encoders=[encoder_1, encoder_2], rho=0)
    optimizer = optim.Adam(dcca_noi.parameters(), lr=1e-3)
    dcca_noi = CCALightning(dcca_noi, optimizer=optimizer)
    trainer = pl.Trainer(
        max_epochs=epochs, log_every_n_steps=10, enable_checkpointing=False
    )
    trainer.fit(dcca_noi, train_loader)
    assert (
            np.testing.assert_array_less(
                cca.score((X, Y)).sum(), trainer.model.score(train_loader).sum()
            )
            is None
    )
    # Soft Decorrelation (stochastic Decorrelation Loss)
    encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
    encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=12)
    sdl = DCCA_SDL(latent_dims, N, encoders=[encoder_1, encoder_2], lam=1e-3)
    optimizer = optim.SGD(sdl.parameters(), lr=1e-1)
    sdl = CCALightning(sdl, optimizer=optimizer)
    trainer = pl.Trainer(max_epochs=epochs, log_every_n_steps=10)
    trainer.fit(sdl, train_loader)
    assert (
            np.testing.assert_array_less(
                cca.score((X, Y)).sum(), trainer.model.score(train_loader).sum()
            )
            is None
    )
    # DCCA
    encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
    encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=12)
    dcca = DCCA(
        latent_dims=latent_dims,
        encoders=[encoder_1, encoder_2],
        objective=objectives.CCA,
    )
    optimizer = optim.SGD(dcca.parameters(), lr=1e-2)
    dcca = CCALightning(dcca, optimizer=optimizer)
    trainer = pl.Trainer(
        max_epochs=epochs, log_every_n_steps=10, enable_checkpointing=False
    )
    trainer.fit(dcca, train_loader)
    assert (
            np.testing.assert_array_less(
                cca.score((X, Y)).sum(), trainer.model.score(train_loader).sum()
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
    optimizer = optim.SGD(dgcca.parameters(), lr=1e-2)
    dgcca = CCALightning(dgcca, optimizer=optimizer)
    trainer = pl.Trainer(
        max_epochs=epochs, log_every_n_steps=10, enable_checkpointing=False
    )
    trainer.fit(dgcca, train_loader)
    assert (
            np.testing.assert_array_less(
                cca.score((X, Y)).sum(), trainer.model.score(train_loader).sum()
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
    optimizer = optim.SGD(dmcca.parameters(), lr=1e-2)
    dmcca = CCALightning(dmcca, optimizer=optimizer)
    trainer = pl.Trainer(
        max_epochs=epochs, log_every_n_steps=10, enable_checkpointing=False
    )
    trainer.fit(dmcca, train_loader)
    assert (
            np.testing.assert_array_less(
                cca.score((X, Y)).sum(), trainer.model.score(train_loader).sum()
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
    optimizer = optim.SGD(barlowtwins.parameters(), lr=1e-2)
    barlowtwins = CCALightning(barlowtwins, optimizer=optimizer)
    trainer = pl.Trainer(
        max_epochs=epochs, log_every_n_steps=10, enable_checkpointing=False
    )
    trainer.fit(barlowtwins, train_loader)
    assert (
            np.testing.assert_array_less(
                cca.score((X, Y)).sum(), trainer.model.score(train_loader).sum()
            )
            is None
    )


def test_DTCCA_methods():
    latent_dims = 2
    epochs = 5
    encoder_1 = architectures.CNNEncoder(latent_dims=10, feature_size=(16, 16))
    encoder_2 = architectures.CNNEncoder(latent_dims=10, feature_size=(16, 16))
    dtcca = DTCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2])
    dtcca = CCALightning(dtcca)
    trainer = pl.Trainer(max_epochs=epochs, enable_checkpointing=False)
    trainer.fit(dtcca, conv_loader)


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
    splitae = CCALightning(splitae)
    trainer = pl.Trainer(max_epochs=5, enable_checkpointing=False)
    trainer.fit(splitae, conv_loader)
    # DCCAE
    dccae = DCCAE(
        latent_dims=latent_dims,
        encoders=[encoder_1, encoder_2],
        decoders=[decoder_1, decoder_2],
    )
    dccae = CCALightning(dccae)
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
    decoder_1 = architectures.Decoder(
        latent_dims=2 * latent_dims, feature_size=10, norm_output=True
    )
    decoder_2 = architectures.Decoder(
        latent_dims=2 * latent_dims, feature_size=12, norm_output=True
    )
    # DVCCA
    dvcca = DVCCA(
        latent_dims=latent_dims,
        encoders=[encoder_1, encoder_2],
        decoders=[decoder_1, decoder_2],
        private_encoders=[private_encoder_1, private_encoder_2],
    )

    dvcca = CCALightning(dvcca)
    trainer = pl.Trainer(max_epochs=5, enable_checkpointing=False)
    trainer.fit(dvcca, train_loader)


def test_DVCCA_methods():
    latent_dims = 2
    encoder_1 = architectures.Encoder(
        latent_dims=latent_dims, feature_size=10, variational=True
    )
    encoder_2 = architectures.Encoder(
        latent_dims=latent_dims, feature_size=12, variational=True
    )
    decoder_1 = architectures.Decoder(
        latent_dims=latent_dims, feature_size=10, norm_output=True
    )
    decoder_2 = architectures.Decoder(
        latent_dims=latent_dims, feature_size=12, norm_output=True
    )
    dvcca = DVCCA(
        latent_dims=latent_dims,
        encoders=[encoder_1, encoder_2],
        decoders=[decoder_1, decoder_2],
    )

    dvcca = CCALightning(dvcca)
    trainer = pl.Trainer(max_epochs=5, enable_checkpointing=False)
    trainer.fit(dvcca, train_loader)


def test_linear():
    encoder_1 = architectures.LinearEncoder(latent_dims=1, feature_size=10)
    encoder_2 = architectures.LinearEncoder(latent_dims=1, feature_size=12)
    dcca = DCCA(latent_dims=1, encoders=[encoder_1, encoder_2])
    optimizer = optim.Adam(dcca.parameters(), lr=1e-1)
    dcca = CCALightning(dcca, optimizer=optimizer)
    trainer = pl.Trainer(max_epochs=50, enable_checkpointing=False)
    trainer.fit(dcca, loader)
    cca = CCA().fit((X, Y))
    # check linear encoder with SGD matches vanilla linear CCA
    assert (
            np.testing.assert_array_almost_equal(
                cca.score((X, Y)), trainer.model.score(loader), decimal=2
            )
            is None
    )
