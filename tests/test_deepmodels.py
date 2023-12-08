import numpy as np
import lightning.pytorch as pl
from lightning import seed_everything
from sklearn.utils.validation import check_random_state
from torch.utils.data import random_split

import cca_zoo.deep._discriminative._dmcca
from cca_zoo.deep import (
    DCCA,
    DCCA_EY,
    DCCA_GHA,
    DCCA_NOI,
    DCCA_SDL,
    DCCA_SVD,
    DCCAE,
    DMCCA,
    DGCCA,
    DVCCA,
    BarlowTwins,
    architectures,
)
from cca_zoo.deep.data import NumpyDataset, get_dataloaders, check_dataset
from cca_zoo.linear import CCA, GCCA, MCCA

seed_everything(0)
rng = check_random_state(0)
feature_size = [3, 4, 5]
X = rng.rand(32, feature_size[0])
Y = rng.rand(32, feature_size[1])
Z = rng.rand(32, feature_size[2])
X -= X.mean(axis=0)
Y -= Y.mean(axis=0)
Z -= Z.mean(axis=0)
X_conv = rng.rand(32, 1, 4, 4)
Y_conv = rng.rand(32, 1, 4, 4)
X_conv -= X_conv.mean(axis=0)
Y_conv -= Y_conv.mean(axis=0)
dataset = NumpyDataset([X, Y, Z])
check_dataset(dataset)
train_dataset, val_dataset = random_split(dataset, [32 - 8, 8])
loader = get_dataloaders(dataset)
train_loader, val_loader = get_dataloaders(train_dataset, val_dataset)
conv_dataset = NumpyDataset((X_conv, Y_conv))
conv_loader = get_dataloaders(conv_dataset)
train_ids = train_dataset.indices
trainer_kwargs = dict(
    enable_checkpointing=False,
    logger=False,
    enable_model_summary=False,
    enable_progress_bar=False,
)


def test_numpy_dataset():
    dataset = NumpyDataset([X, Y, Z])
    check_dataset(dataset)
    get_dataloaders(dataset)


def test_linear_mcca():
    max_epochs = 50
    latent_dimensions = 2
    mcca = MCCA(latent_dimensions=latent_dimensions).fit((X, Y, Z))
    # DMCCA
    encoder_1 = architectures.LinearEncoder(
        latent_dimensions=latent_dimensions, feature_size=feature_size[0]
    )
    encoder_2 = architectures.LinearEncoder(
        latent_dimensions=latent_dimensions, feature_size=feature_size[1]
    )
    encoder_3 = architectures.LinearEncoder(
        latent_dimensions=latent_dimensions, feature_size=feature_size[2]
    )
    dmcca = DMCCA(
        latent_dimensions=latent_dimensions,
        encoders=[encoder_1, encoder_2, encoder_3],
        lr=1e-1,
    )
    trainer = pl.Trainer(max_epochs=max_epochs, **trainer_kwargs)
    trainer.fit(dmcca, loader)
    assert (
        np.testing.assert_array_almost_equal(
            mcca.score((X, Y, Z)), dmcca.score(loader), decimal=2
        )
        is None
    )


def test_linear_gcca():
    max_epochs = 50
    latent_dimensions = 2
    gcca = GCCA(latent_dimensions=latent_dimensions).fit((X, Y, Z))
    # DCCA_GCCA
    encoder_1 = architectures.LinearEncoder(
        latent_dimensions=latent_dimensions, feature_size=feature_size[0]
    )
    encoder_2 = architectures.LinearEncoder(
        latent_dimensions=latent_dimensions, feature_size=feature_size[1]
    )
    encoder_3 = architectures.LinearEncoder(
        latent_dimensions=latent_dimensions, feature_size=feature_size[2]
    )
    dgcca = DGCCA(
        latent_dimensions=latent_dimensions,
        encoders=[encoder_1, encoder_2, encoder_3],
        lr=1e-2,
    )
    trainer = pl.Trainer(max_epochs=max_epochs, **trainer_kwargs)
    trainer.fit(dgcca, loader)
    assert (
        np.testing.assert_array_almost_equal(
            gcca.score((X, Y, Z)).sum(), dgcca.score(loader), decimal=1
        )
        is None
    )


def test_DCCA_methods():
    max_epochs = 50
    latent_dimensions = 2
    cca = CCA(latent_dimensions=latent_dimensions).fit((X, Y))
    # DCCA
    encoder_1 = architectures.Encoder(
        latent_dimensions=latent_dimensions, feature_size=feature_size[0]
    )
    encoder_2 = architectures.Encoder(
        latent_dimensions=latent_dimensions, feature_size=feature_size[1]
    )
    dcca = DCCA(
        latent_dimensions=latent_dimensions,
        encoders=[encoder_1, encoder_2],
        lr=1e-2,
    )
    trainer = pl.Trainer(max_epochs=max_epochs, **trainer_kwargs)
    trainer.fit(dcca, train_loader, val_dataloaders=val_loader)
    assert (
        np.testing.assert_array_less(cca.score((X, Y)), dcca.score(train_loader))
        is None
    )
    # DMCCA
    encoder_1 = architectures.Encoder(
        latent_dimensions=latent_dimensions, feature_size=feature_size[0]
    )
    encoder_2 = architectures.Encoder(
        latent_dimensions=latent_dimensions, feature_size=feature_size[1]
    )
    dmcca = DMCCA(
        latent_dimensions=latent_dimensions,
        encoders=[encoder_1, encoder_2],
        lr=1e-3,
    )
    trainer = pl.Trainer(max_epochs=max_epochs, **trainer_kwargs)
    trainer.fit(dmcca, train_loader, val_dataloaders=val_loader)
    assert (
        np.testing.assert_array_less(cca.score((X, Y)), dmcca.score(train_loader))
        is None
    )
    # DCCA_GHA
    encoder_1 = architectures.Encoder(
        latent_dimensions=latent_dimensions, feature_size=feature_size[0]
    )
    encoder_2 = architectures.Encoder(
        latent_dimensions=latent_dimensions, feature_size=feature_size[1]
    )
    dcca_gha = DCCA_GHA(
        latent_dimensions=latent_dimensions,
        encoders=[encoder_1, encoder_2],
    )
    trainer = pl.Trainer(max_epochs=max_epochs, **trainer_kwargs)
    trainer.fit(dcca_gha, train_loader, val_dataloaders=val_loader)
    assert (
        np.testing.assert_array_less(cca.score((X, Y)), dcca_gha.score(train_loader))
        is None
    )
    # DCCA_SVD
    encoder_1 = architectures.Encoder(
        latent_dimensions=latent_dimensions, feature_size=feature_size[0]
    )
    encoder_2 = architectures.Encoder(
        latent_dimensions=latent_dimensions, feature_size=feature_size[1]
    )
    dcca_svd = DCCA_SVD(
        latent_dimensions=latent_dimensions,
        encoders=[encoder_1, encoder_2],
    )
    trainer = pl.Trainer(max_epochs=max_epochs, **trainer_kwargs)
    trainer.fit(dcca_svd, train_loader, val_dataloaders=val_loader)
    assert (
        np.testing.assert_array_less(cca.score((X, Y)), dcca_svd.score(train_loader))
        is None
    )
    # DCCA_EY
    encoder_1 = architectures.Encoder(
        latent_dimensions=latent_dimensions, feature_size=feature_size[0]
    )
    encoder_2 = architectures.Encoder(
        latent_dimensions=latent_dimensions, feature_size=feature_size[1]
    )
    dcca_ey = DCCA_EY(
        latent_dimensions=latent_dimensions,
        encoders=[encoder_1, encoder_2],

    )
    trainer = pl.Trainer(max_epochs=max_epochs, **trainer_kwargs)
    trainer.fit(dcca_ey, train_loader, val_dataloaders=val_loader)
    assert (
        np.testing.assert_array_less(cca.score((X, Y)), dcca_ey.score(train_loader))
        is None
    )
    # DCCA_NOI
    encoder_1 = architectures.Encoder(
        latent_dimensions=latent_dimensions, feature_size=feature_size[0]
    )
    encoder_2 = architectures.Encoder(
        latent_dimensions=latent_dimensions, feature_size=feature_size[1]
    )
    dcca_noi = DCCA_NOI(
        latent_dimensions, encoders=[encoder_1, encoder_2], rho=0.2, lr=1e-1
    )
    trainer = pl.Trainer(max_epochs=max_epochs, **trainer_kwargs)
    trainer.fit(dcca_noi, train_loader)
    assert (
        np.testing.assert_array_less(cca.score((X, Y)), dcca_noi.score(train_loader))
        is None
    )
    # Soft Decorrelation (_stochastic Decorrelation Loss)
    encoder_1 = architectures.Encoder(
        latent_dimensions=latent_dimensions, feature_size=feature_size[0]
    )
    encoder_2 = architectures.Encoder(
        latent_dimensions=latent_dimensions, feature_size=feature_size[1]
    )
    sdl = DCCA_SDL(
        latent_dimensions, encoders=[encoder_1, encoder_2], lam=1e-2, lr=1e-3
    )
    trainer = pl.Trainer(max_epochs=max_epochs, **trainer_kwargs)
    trainer.fit(sdl, train_loader)
    assert (
        np.testing.assert_array_less(cca.score((X, Y)), sdl.score(train_loader)) is None
    )
    # Barlow Twins
    encoder_1 = architectures.Encoder(
        latent_dimensions=latent_dimensions, feature_size=feature_size[0]
    )
    encoder_2 = architectures.Encoder(
        latent_dimensions=latent_dimensions, feature_size=feature_size[1]
    )
    barlowtwins = BarlowTwins(
        latent_dimensions=latent_dimensions,
        encoders=[encoder_1, encoder_2],
    )
    trainer = pl.Trainer(max_epochs=max_epochs, **trainer_kwargs)
    trainer.fit(barlowtwins, train_loader)
    assert (
        np.testing.assert_array_less(cca.score((X, Y)), barlowtwins.score(train_loader))
        is None
    )
    # DGCCA
    encoder_1 = architectures.Encoder(
        latent_dimensions=latent_dimensions, feature_size=feature_size[0]
    )
    encoder_2 = architectures.Encoder(
        latent_dimensions=latent_dimensions, feature_size=feature_size[1]
    )
    dgcca = DGCCA(
        latent_dimensions=latent_dimensions,
        encoders=[encoder_1, encoder_2],
    )
    trainer = pl.Trainer(max_epochs=max_epochs, **trainer_kwargs)
    trainer.fit(dgcca, train_loader)
    assert (
        np.testing.assert_array_less(cca.score((X, Y)), dgcca.score(train_loader))
        is None
    )
    # DMCCA
    encoder_1 = architectures.Encoder(
        latent_dimensions=latent_dimensions, feature_size=feature_size[0]
    )
    encoder_2 = architectures.Encoder(
        latent_dimensions=latent_dimensions, feature_size=feature_size[1]
    )
    dmcca = DMCCA(
        latent_dimensions=latent_dimensions,
        encoders=[encoder_1, encoder_2],
    )
    trainer = pl.Trainer(max_epochs=max_epochs, **trainer_kwargs)
    trainer.fit(dmcca, train_loader)
    assert (
        np.testing.assert_array_less(cca.score((X, Y)), dmcca.score(train_loader))
        is None
    )


def test_DCCAE_methods():
    max_epochs = 2
    latent_dimensions = 2
    encoder_1 = architectures.CNNEncoder(
        latent_dimensions=latent_dimensions, feature_size=(4, 4)
    )
    encoder_2 = architectures.CNNEncoder(
        latent_dimensions=latent_dimensions, feature_size=(4, 4)
    )
    decoder_1 = architectures.CNNDecoder(
        latent_dimensions=latent_dimensions, feature_size=(4, 4)
    )
    decoder_2 = architectures.CNNDecoder(
        latent_dimensions=latent_dimensions, feature_size=(4, 4)
    )
    # DCCAE
    dccae = DCCAE(
        latent_dimensions=latent_dimensions,
        encoders=[encoder_1, encoder_2],
        decoders=[decoder_1, decoder_2],
    )
    trainer = pl.Trainer(max_epochs=max_epochs, **trainer_kwargs)
    trainer.fit(dccae, conv_loader)


def test_DVCCA_p_methods():
    max_epochs = 2
    latent_dimensions = 2
    encoder_1 = architectures.Encoder(
        latent_dimensions=latent_dimensions,
        feature_size=feature_size[0],
        variational=True,
    )
    encoder_2 = architectures.Encoder(
        latent_dimensions=latent_dimensions,
        feature_size=feature_size[1],
        variational=True,
    )
    private_encoder_1 = architectures.Encoder(
        latent_dimensions=latent_dimensions,
        feature_size=feature_size[0],
        variational=True,
    )
    private_encoder_2 = architectures.Encoder(
        latent_dimensions=latent_dimensions,
        feature_size=feature_size[1],
        variational=True,
    )
    decoder_1 = architectures.Decoder(
        latent_dimensions=2 * latent_dimensions, feature_size=feature_size[0]
    )
    decoder_2 = architectures.Decoder(
        latent_dimensions=2 * latent_dimensions, feature_size=feature_size[1]
    )
    # DVCCA
    dvcca = DVCCA(
        latent_dimensions=latent_dimensions,
        encoders=[encoder_1, encoder_2],
        decoders=[decoder_1, decoder_2],
        private_encoders=[private_encoder_1, private_encoder_2],
    )
    trainer = pl.Trainer(max_epochs=max_epochs, **trainer_kwargs)
    trainer.fit(dvcca, train_loader)
    dvcca.transform(train_loader)


def test_DVCCA_methods():
    max_epochs = 2
    latent_dimensions = 2
    encoder_1 = architectures.Encoder(
        latent_dimensions=latent_dimensions,
        feature_size=feature_size[0],
        variational=True,
    )
    encoder_2 = architectures.Encoder(
        latent_dimensions=latent_dimensions,
        feature_size=feature_size[1],
        variational=True,
    )
    decoder_1 = architectures.Decoder(
        latent_dimensions=latent_dimensions, feature_size=feature_size[0]
    )
    decoder_2 = architectures.Decoder(
        latent_dimensions=latent_dimensions, feature_size=feature_size[1]
    )
    dvcca = DVCCA(
        latent_dimensions=latent_dimensions,
        encoders=[encoder_1, encoder_2],
        decoders=[decoder_1, decoder_2],
    )
    trainer = pl.Trainer(max_epochs=max_epochs, **trainer_kwargs)
    trainer.fit(dvcca, train_loader)
