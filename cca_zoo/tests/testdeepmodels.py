from unittest import TestCase

import numpy as np
from sklearn.utils.validation import check_random_state
from torch import optim, manual_seed
from torch.utils.data import Subset

from cca_zoo import data
from cca_zoo.data import Noisy_MNIST_Dataset
from cca_zoo.deepmodels import DCCA, DCCAE, DVCCA, DCCA_NOI, DTCCA, SplitAE, DeepWrapper
from cca_zoo.deepmodels import objectives, architectures
from cca_zoo.models import CCA


class TestDeepModels(TestCase):

    def setUp(self):
        manual_seed(0)
        self.rng = check_random_state(0)
        self.X = self.rng.rand(200, 10)
        self.Y = self.rng.rand(200, 10)
        self.Z = self.rng.rand(200, 10)
        self.X_conv = self.rng.rand(100, 1, 16, 16)
        self.Y_conv = self.rng.rand(100, 1, 16, 16)
        self.train_dataset = data.CCA_Dataset(self.X, self.Y)

    def tearDown(self):
        pass

    def test_input_types(self):
        latent_dims = 2
        device = 'cpu'
        encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
        encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
        # DCCA
        dcca_model = DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2],
                          objective=objectives.CCA)
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dcca_model = DeepWrapper(dcca_model, device=device)
        dcca_model.fit(self.train_dataset, epochs=3)
        dcca_model.fit(self.train_dataset, val_dataset=self.train_dataset, epochs=3)
        dcca_model.fit((self.X, self.Y), val_dataset=(self.X, self.Y), epochs=3)
        dcca_model.fit((self.X, self.Y), val_split=0.2, epochs=3)

    def tutorial_test(self):
        # Load MNIST Data
        N = 500
        latent_dims = 2
        dataset = Noisy_MNIST_Dataset(mnist_type='FashionMNIST', train=True)
        ids = np.arange(min(2 * N, len(dataset)))
        np.random.shuffle(ids)
        train_ids, val_ids = np.array_split(ids, 2)
        val_dataset = Subset(dataset, val_ids)
        train_dataset = Subset(dataset, train_ids)
        test_dataset = Noisy_MNIST_Dataset(mnist_type='FashionMNIST', train=False)
        test_ids = np.arange(min(N, len(test_dataset)))
        np.random.shuffle(test_ids)
        test_dataset = Subset(test_dataset, test_ids)
        print('DCCA')
        encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=784)
        encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=784)
        dcca_model = DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2])
        dcca_model = DeepWrapper(dcca_model)
        dcca_model.fit(train_dataset, val_dataset=val_dataset, epochs=2)
        dcca_results = np.stack((dcca_model.score(train_dataset), dcca_model.correlations(test_dataset)[0, 1]))

    def test_large_p(self):
        large_p = 256
        X = self.rng.rand(2000, large_p)
        Y = self.rng.rand(2000, large_p)
        latent_dims = 32
        device = 'cpu'
        encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=large_p)
        encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=large_p)
        dcca_model = DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2],
                          objective=objectives.MCCA, eps=1e-3).float()
        optimizer = optim.Adam(dcca_model.parameters(), lr=1e-4)
        dcca_model = DeepWrapper(dcca_model, device=device, optimizer=optimizer)
        dcca_model.fit((X, Y), epochs=100)
        cca_model = CCA(latent_dims=latent_dims).fit(X, Y)

    def test_DCCA_methods_cpu(self):
        latent_dims = 4
        cca_model = CCA(latent_dims=latent_dims).fit(self.X, self.Y)
        device = 'cpu'
        epochs = 100
        # DCCA
        encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
        encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
        dcca_model = DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2],
                          objective=objectives.CCA)
        optimizer = optim.SGD(dcca_model.parameters(), lr=1e-1)
        dcca_model = DeepWrapper(dcca_model, device=device, optimizer=optimizer)
        dcca_model.fit((self.X, self.Y), epochs=epochs)
        self.assertIsNone(
            np.testing.assert_array_less(cca_model.score(self.X, self.Y).sum(),
                                         dcca_model.score((self.X, self.Y)).sum()))
        # DGCCA
        encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
        encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
        dgcca_model = DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2],
                           objective=objectives.GCCA)
        optimizer = optim.SGD(dgcca_model.parameters(), lr=1e-1)
        dgcca_model = DeepWrapper(dgcca_model, device=device, optimizer=optimizer)
        dgcca_model.fit((self.X, self.Y), epochs=epochs)
        self.assertIsNone(
            np.testing.assert_array_less(cca_model.score(self.X, self.Y).sum(),
                                         dgcca_model.score((self.X, self.Y)).sum()))
        # DMCCA
        encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
        encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
        dmcca_model = DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2],
                           objective=objectives.MCCA)
        optimizer = optim.SGD(dmcca_model.parameters(), lr=1e-1)
        dmcca_model = DeepWrapper(dmcca_model, device=device, optimizer=optimizer)
        dmcca_model.fit((self.X, self.Y), epochs=epochs)
        self.assertIsNone(
            np.testing.assert_array_less(cca_model.score(self.X, self.Y).sum(),
                                         dmcca_model.score((self.X, self.Y)).sum()))
        # DCCA_NOI
        encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
        encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
        dcca_noi_model = DCCA_NOI(latent_dims, self.X.shape[0], encoders=[encoder_1, encoder_2], rho=0.5)
        optimizer = optim.Adam(dcca_noi_model.parameters(), lr=1e-3)
        dcca_noi_model = DeepWrapper(dcca_noi_model, device=device, optimizer=optimizer)
        dcca_noi_model.fit((self.X, self.Y), epochs=epochs)
        self.assertIsNone(
            np.testing.assert_array_less(cca_model.score(self.X, self.Y).sum(),
                                         dcca_noi_model.score((self.X, self.Y)).sum()))

    def test_DTCCA_methods_cpu(self):
        latent_dims = 2
        device = 'cpu'
        encoder_1 = architectures.Encoder(latent_dims=10, feature_size=10)
        encoder_2 = architectures.Encoder(latent_dims=10, feature_size=10)
        encoder_3 = architectures.Encoder(latent_dims=10, feature_size=10)
        # DTCCA
        dtcca_model = DTCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2])
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dtcca_model = DeepWrapper(dtcca_model, device=device)
        dtcca_model.fit((self.X, self.Y), epochs=20)
        # DCCA
        dcca_model = DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2],
                          objective=objectives.GCCA)
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dcca_model = DeepWrapper(dcca_model, device=device)
        dcca_model.fit((self.X, self.Y), epochs=20)

    def test_scheduler(self):
        latent_dims = 2
        device = 'cpu'
        encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
        encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
        # DCCA
        dcca_model = DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2],
                          objective=objectives.CCA)
        optimizer = optim.Adam(dcca_model.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 1)
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dcca_model = DeepWrapper(dcca_model, device=device, optimizer=optimizer, scheduler=scheduler)
        dcca_model.fit((self.X, self.Y), epochs=20)

    def test_DGCCA_methods_cpu(self):
        latent_dims = 2
        device = 'cpu'
        encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
        encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
        encoder_3 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
        # DTCCA
        dtcca_model = DTCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2])
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dtcca_model = DeepWrapper(dtcca_model, device=device)
        dtcca_model.fit((self.X, self.Y, self.Z))
        # DGCCA
        dgcca_model = DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2, encoder_3],
                           objective=objectives.GCCA)
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dgcca_model = DeepWrapper(dgcca_model, device=device)
        dgcca_model.fit((self.X, self.Y, self.Z))
        # DMCCA
        dmcca_model = DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2, encoder_3],
                           objective=objectives.MCCA)
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dmcca_model = DeepWrapper(dmcca_model, device=device)
        dmcca_model.fit((self.X, self.Y, self.Z))

    def test_DCCAE_methods_cpu(self):
        latent_dims = 2
        device = 'cpu'
        encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
        encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
        decoder_1 = architectures.Decoder(latent_dims=latent_dims, feature_size=10)
        decoder_2 = architectures.Decoder(latent_dims=latent_dims, feature_size=10)
        # DCCAE
        dccae_model = DCCAE(latent_dims=latent_dims, encoders=[encoder_1, encoder_2],
                            decoders=[decoder_1, decoder_2])
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dccae_model = DeepWrapper(dccae_model, device=device)
        dccae_model.fit((self.X, self.Y), epochs=20)
        # SplitAE
        splitae_model = SplitAE(latent_dims=latent_dims, encoder=encoder_1,
                                decoders=[decoder_1, decoder_2])
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        splitae_model = DeepWrapper(splitae_model, device=device)
        splitae_model.fit((self.X, self.Y), epochs=10)

    def test_DCCAEconv_methods_cpu(self):
        latent_dims = 2
        device = 'cpu'
        encoder_1 = architectures.CNNEncoder(latent_dims=latent_dims, feature_size=[16, 16])
        encoder_2 = architectures.CNNEncoder(latent_dims=latent_dims, feature_size=[16, 16])
        decoder_1 = architectures.CNNDecoder(latent_dims=latent_dims, feature_size=[16, 16])
        decoder_2 = architectures.CNNDecoder(latent_dims=latent_dims, feature_size=[16, 16])
        # DCCAE
        dccae_model = DCCAE(latent_dims=latent_dims, encoders=[encoder_1, encoder_2],
                            decoders=[decoder_1, decoder_2])
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dccae_model = DeepWrapper(dccae_model, device=device)
        dccae_model.fit((self.X_conv, self.Y_conv))

    def test_DVCCA_methods_cpu(self):
        latent_dims = 2
        device = 'cpu'
        encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=10, variational=True)
        encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=10, variational=True)
        decoder_1 = architectures.Decoder(latent_dims=latent_dims, feature_size=10, norm_output=True)
        decoder_2 = architectures.Decoder(latent_dims=latent_dims, feature_size=10, norm_output=True)
        # DVCCA
        dvcca_model = DVCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2],
                            decoders=[decoder_1, decoder_2])
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dvcca_model = DeepWrapper(dvcca_model, device=device)
        dvcca_model.fit((self.X, self.Y))

    def test_DVCCA_p_methods_cpu(self):
        latent_dims = 2
        device = 'cpu'
        encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=10, variational=True)
        encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=10, variational=True)
        private_encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=10, variational=True)
        private_encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=10, variational=True)
        decoder_1 = architectures.Decoder(latent_dims=2 * latent_dims, feature_size=10, norm_output=True)
        decoder_2 = architectures.Decoder(latent_dims=2 * latent_dims, feature_size=10, norm_output=True)
        # DVCCA
        dvcca_model = DVCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2],
                            decoders=[decoder_1, decoder_2],
                            private_encoders=[private_encoder_1, private_encoder_2])
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dvcca_model = DeepWrapper(dvcca_model, device=device)
        dvcca_model.fit((self.X, self.Y))

    def test_DCCA_methods_gpu(self):
        latent_dims = 2
        device = 'cuda'
        encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
        encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
        # DCCA
        dcca_model = DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2],
                          objective=objectives.CCA)
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dcca_model = DeepWrapper(dcca_model, device=device)
        dcca_model.fit((self.X, self.Y))
        # DGCCA
        dgcca_model = DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2],
                           objective=objectives.GCCA)
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dgcca_model = DeepWrapper(dgcca_model, device=device)
        dgcca_model.fit((self.X, self.Y))
        # DMCCA
        dmcca_model = DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2],
                           objective=objectives.MCCA)
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dmcca_model = DeepWrapper(dmcca_model, device=device)
        dmcca_model.fit((self.X, self.Y))
        # DCCA_NOI
        dcca_noi_model = DCCA_NOI(latent_dims, self.X.shape[0], encoders=[encoder_1, encoder_2])
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dcca_noi_model = DeepWrapper(dcca_noi_model, device=device)
        dcca_noi_model.fit((self.X, self.Y))

    def test_DGCCA_methods_gpu(self):
        latent_dims = 2
        device = 'cuda'
        encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
        encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
        encoder_3 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
        # DGCCA
        dgcca_model = DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2, encoder_3],
                           objective=objectives.GCCA)
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dgcca_model = DeepWrapper(dgcca_model, device=device)
        dgcca_model.fit((self.X, self.Y, self.Z))
        # DMCCA
        dmcca_model = DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2, encoder_3],
                           objective=objectives.MCCA)
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dmcca_model = DeepWrapper(dmcca_model, device=device)
        dmcca_model.fit((self.X, self.Y, self.Z))

    def test_DCCAE_methods_gpu(self):
        latent_dims = 2
        device = 'cuda'
        encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
        encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
        decoder_1 = architectures.Decoder(latent_dims=latent_dims, feature_size=10)
        decoder_2 = architectures.Decoder(latent_dims=latent_dims, feature_size=10)
        # DCCAE
        dccae_model = DCCAE(latent_dims=latent_dims, encoders=[encoder_1, encoder_2],
                            decoders=[decoder_1, decoder_2])
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dccae_model = DeepWrapper(dccae_model, device=device)
        dccae_model.fit((self.X, self.Y))

    def test_DVCCA_methods_gpu(self):
        latent_dims = 2
        device = 'cuda'
        encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=10, variational=True)
        encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=10, variational=True)
        decoder_1 = architectures.Decoder(latent_dims=latent_dims, feature_size=10, norm_output=True)
        decoder_2 = architectures.Decoder(latent_dims=latent_dims, feature_size=10, norm_output=True)
        # DVCCA
        dvcca_model = DVCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2],
                            decoders=[decoder_1, decoder_2])
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dvcca_model = DeepWrapper(dvcca_model, device=device)
        dvcca_model.fit((self.X, self.Y))
