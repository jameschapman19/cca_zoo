import numpy as np
import pytorch_lightning as pl
import torch.utils.data
from sklearn.utils.validation import check_random_state
from torch import optim, manual_seed
from torch.utils.data import Subset

from cca_zoo import data
from cca_zoo.data import Noisy_MNIST_Dataset
from cca_zoo.deepmodels import DCCA, DCCAE, DVCCA, DCCA_NOI, DTCCA, SplitAE, CCALightning, get_dataloaders, \
    process_data
from cca_zoo.deepmodels import objectives, architectures
from cca_zoo.models import CCA

manual_seed(0)
rng = check_random_state(0)
X = rng.rand(200, 10)
Y = rng.rand(200, 12)
Z = rng.rand(200, 14)
X_conv = rng.rand(100, 1, 16, 16)
Y_conv = rng.rand(100, 1, 16, 16)
dataset = data.CCA_Dataset([X, Y, Z])
train_dataset, val_dataset = process_data(
    dataset,
    val_split=0.2)
train_dataset_numpy, val_dataset_numpy = process_data(
    (X, Y, Z),
    val_split=0.2)
loader = get_dataloaders(dataset)
train_loader, val_loader = get_dataloaders(train_dataset, val_dataset)
train_loader_numpy, val_loader_numpy = get_dataloaders(train_dataset, val_dataset)
conv_dataset = data.CCA_Dataset((X_conv, Y_conv))
conv_loader = get_dataloaders(conv_dataset)


def test_input_types():
    latent_dims = 2
    encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
    encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=12)
    # DCCA
    dcca_model = DCCA(
        latent_dims=latent_dims,
        encoders=[encoder_1, encoder_2],
        objective=objectives.CCA,
    )
    dcca_model = CCALightning(dcca_model)
    trainer = pl.Trainer(gpus=0, max_epochs=5, progress_bar_refresh_rate=1, log_every_n_steps=1, logger=False)
    trainer.fit(dcca_model, train_loader, val_loader)
    trainer_numpy = pl.Trainer(gpus=0, max_epochs=5, progress_bar_refresh_rate=1, log_every_n_steps=1, logger=False)
    trainer_numpy.fit(dcca_model, train_loader_numpy, val_loader_numpy)


def test_tutorial():
    N = 500
    latent_dims = 2
    dataset = Noisy_MNIST_Dataset(mnist_type="FashionMNIST", train=True)
    ids = np.arange(min(2 * N, len(dataset)))
    np.random.shuffle(ids)
    train_ids, val_ids = np.array_split(ids, 2)
    val_dataset = Subset(dataset, val_ids)
    train_dataset = Subset(dataset, train_ids)
    test_dataset = Noisy_MNIST_Dataset(mnist_type="FashionMNIST", train=False)
    test_ids = np.arange(min(N, len(test_dataset)))
    np.random.shuffle(test_ids)
    test_dataset = Subset(test_dataset, test_ids)
    encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=784)
    encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=784)
    train_loader, val_loader = get_dataloaders(train_dataset, val_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))
    dcca_model = DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2])
    dcca_model = CCALightning(dcca_model)
    trainer = pl.Trainer(gpus=0, max_epochs=5, progress_bar_refresh_rate=1, log_every_n_steps=1, logger=False)
    trainer.fit(dcca_model, train_loader, val_loader)
    dcca_results = np.stack(
        (dcca_model.score(train_loader), dcca_model.correlations(test_loader)[0, 1])
    )


def test_large_p():
    large_p = 256
    X = rng.rand(2000, large_p)
    Y = rng.rand(2000, large_p)
    dataset = data.CCA_Dataset([X, Y])
    loader = get_dataloaders(dataset)
    latent_dims = 32
    encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=large_p)
    encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=large_p)
    dcca_model = DCCA(
        latent_dims=latent_dims,
        encoders=[encoder_1, encoder_2],
        objective=objectives.MCCA,
        eps=1e-3,
    )
    optimizer = optim.Adam(dcca_model.parameters(), lr=1e-4)
    dcca_model = CCALightning(dcca_model, optimizer=optimizer)
    trainer = pl.Trainer(max_epochs=5, progress_bar_refresh_rate=1, log_every_n_steps=1, logger=False)
    trainer.fit(dcca_model, loader)


def test_DCCA_methods():
    latent_dims = 2
    epochs = 100
    cca_model = CCA(latent_dims=latent_dims).fit((X, Y))
    # DCCA_NOI
    encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
    encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=12)
    dcca_noi_model = DCCA_NOI(
        latent_dims, X.shape[0], encoders=[encoder_1, encoder_2], rho=0
    )
    optimizer = optim.Adam(dcca_noi_model.parameters(), lr=1e-2)
    dcca_noi_model = CCALightning(dcca_noi_model, optimizer=optimizer)
    trainer = pl.Trainer(max_epochs=epochs, progress_bar_refresh_rate=1, log_every_n_steps=1, logger=False)
    trainer.fit(dcca_noi_model, train_loader)
    assert (
            np.testing.assert_array_less(
                cca_model.score((X, Y)).sum(), trainer.model.score(train_loader).sum()
            )
            is None
    )
    # DCCA
    encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
    encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=12)
    dcca_model = DCCA(
        latent_dims=latent_dims,
        encoders=[encoder_1, encoder_2],
        objective=objectives.CCA,
    )
    optimizer = optim.SGD(dcca_model.parameters(), lr=1e-2)
    dcca_model = CCALightning(dcca_model, optimizer=optimizer)
    trainer = pl.Trainer(max_epochs=epochs, progress_bar_refresh_rate=1, log_every_n_steps=1, logger=False)
    trainer.fit(dcca_model, train_loader)
    assert (
            np.testing.assert_array_less(
                cca_model.score((X, Y)).sum(), trainer.model.score(train_loader).sum()
            )
            is None
    )
    # DGCCA
    encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
    encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=12)
    dgcca_model = DCCA(
        latent_dims=latent_dims,
        encoders=[encoder_1, encoder_2],
        objective=objectives.GCCA,
    )
    optimizer = optim.SGD(dgcca_model.parameters(), lr=1e-2)
    dgcca_model = CCALightning(dgcca_model, optimizer=optimizer)
    trainer = pl.Trainer(max_epochs=epochs, progress_bar_refresh_rate=1, log_every_n_steps=1, logger=False)
    trainer.fit(dgcca_model, train_loader)
    assert (
            np.testing.assert_array_less(
                cca_model.score((X, Y)).sum(), trainer.model.score(train_loader).sum()
            )
            is None
    )
    # DMCCA
    encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
    encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=12)
    dmcca_model = DCCA(
        latent_dims=latent_dims,
        encoders=[encoder_1, encoder_2],
        objective=objectives.MCCA,
    )
    optimizer = optim.SGD(dmcca_model.parameters(), lr=1e-2)
    dmcca_model = CCALightning(dmcca_model, optimizer=optimizer)
    trainer = pl.Trainer(max_epochs=epochs, progress_bar_refresh_rate=1, log_every_n_steps=1, logger=False)
    trainer.fit(dmcca_model, train_loader)
    assert (
            np.testing.assert_array_less(
                cca_model.score((X, Y)).sum(), trainer.model.score(train_loader).sum()
            )
            is None
    )


def test_DTCCA_methods():
    latent_dims = 2
    epochs = 5
    encoder_1 = architectures.Encoder(latent_dims=10, feature_size=10)
    encoder_2 = architectures.Encoder(latent_dims=10, feature_size=12)
    dtcca_model = DTCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2])
    dtcca_model = CCALightning(dtcca_model)
    trainer = pl.Trainer(max_epochs=epochs, progress_bar_refresh_rate=1, log_every_n_steps=1, logger=False)
    trainer.fit(dtcca_model, train_loader)


def test_scheduler():
    latent_dims = 2
    encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
    encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=12)
    dcca_model = DCCA(
        latent_dims=latent_dims,
        encoders=[encoder_1, encoder_2],
        objective=objectives.CCA,
    )
    optimizer = optim.Adam(dcca_model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 1)
    dcca_model = CCALightning(dcca_model, optimizer=optimizer, lr_scheduler=scheduler)
    trainer = pl.Trainer(max_epochs=5, progress_bar_refresh_rate=1, log_every_n_steps=1, logger=False)
    trainer.fit(dcca_model, train_loader)


def test_DCCAE_methods():
    latent_dims = 2
    encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
    encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=12)
    decoder_1 = architectures.Decoder(latent_dims=latent_dims, feature_size=10)
    decoder_2 = architectures.Decoder(latent_dims=latent_dims, feature_size=12)
    # SplitAE
    splitae_model = SplitAE(
        latent_dims=latent_dims, encoder=encoder_1, decoders=[decoder_1, decoder_2]
    )
    splitae_model = CCALightning(splitae_model)
    trainer = pl.Trainer(max_epochs=5, progress_bar_refresh_rate=1, log_every_n_steps=1, logger=False)
    trainer.fit(splitae_model, train_loader)
    # DCCAE
    dccae_model = DCCAE(
        latent_dims=latent_dims,
        encoders=[encoder_1, encoder_2],
        decoders=[decoder_1, decoder_2],
    )
    dccae_model = CCALightning(dccae_model)
    trainer = pl.Trainer(max_epochs=5, progress_bar_refresh_rate=1, log_every_n_steps=1, logger=False)
    trainer.fit(dccae_model, train_loader)


def test_DCCAEconv_methods():
    latent_dims = 2
    encoder_1 = architectures.CNNEncoder(latent_dims=latent_dims, feature_size=[16, 16])
    encoder_2 = architectures.CNNEncoder(latent_dims=latent_dims, feature_size=[16, 16])
    decoder_1 = architectures.CNNDecoder(latent_dims=latent_dims, feature_size=[16, 16])
    decoder_2 = architectures.CNNDecoder(latent_dims=latent_dims, feature_size=[16, 16])
    # DCCAE
    dccae_model = DCCAE(
        latent_dims=latent_dims,
        encoders=[encoder_1, encoder_2],
        decoders=[decoder_1, decoder_2],
    )
    dccae_model = CCALightning(dccae_model)
    trainer = pl.Trainer(max_epochs=5, progress_bar_refresh_rate=1, log_every_n_steps=1, logger=False)
    trainer.fit(dccae_model, conv_loader)


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
    dvcca_model = DVCCA(
        latent_dims=latent_dims,
        encoders=[encoder_1, encoder_2],
        decoders=[decoder_1, decoder_2],
        private_encoders=[private_encoder_1, private_encoder_2],
    )

    dvcca_model = CCALightning(dvcca_model)
    trainer = pl.Trainer(max_epochs=5, progress_bar_refresh_rate=1, log_every_n_steps=1, logger=False)
    trainer.fit(dvcca_model, train_loader)


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
    dvcca_model = DVCCA(
        latent_dims=latent_dims,
        encoders=[encoder_1, encoder_2],
        decoders=[decoder_1, decoder_2],
    )

    dvcca_model = CCALightning(dvcca_model)
    trainer = pl.Trainer(max_epochs=5, progress_bar_refresh_rate=1, log_every_n_steps=1, logger=False)
    trainer.fit(dvcca_model, train_loader)


def test_linear():
    encoder_1 = architectures.LinearEncoder(latent_dims=1, feature_size=10)
    encoder_2 = architectures.LinearEncoder(latent_dims=1, feature_size=12)
    dcca_model = DCCA(latent_dims=1, encoders=[encoder_1, encoder_2])
    dcca_model = CCALightning(dcca_model, learning_rate=1e-1)
    trainer = pl.Trainer(gpus=0, max_epochs=50, progress_bar_refresh_rate=1, log_every_n_steps=1, logger=False)
    trainer.fit(dcca_model, loader)
    cca = CCA().fit((X, Y))
    # check linear encoder with SGD matches vanilla linear CCA
    assert (
            np.testing.assert_array_almost_equal(
                cca.score((X, Y)), trainer.model.score(loader), decimal=2
            )
            is None
    )
