from unittest import TestCase

import numpy as np
from torch import optim

from cca_zoo import data
from cca_zoo.deepmodels import DCCA, DCCAE, DVCCA, DCCA_NOI, DTCCA, SplitAE, DeepWrapper
from cca_zoo.deepmodels import objectives, architectures


class TestDeepWrapper(TestCase):

    def setUp(self):
        self.X = np.random.rand(200, 10)
        self.Y = np.random.rand(200, 10)
        self.Z = np.random.rand(200, 10)
        self.X_conv = np.random.rand(100, 1, 16, 16)
        self.Y_conv = np.random.rand(100, 1, 16, 16)
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

    def test_DCCA_methods_cpu(self):
        latent_dims = 2
        device = 'cpu'
        encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
        encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
        # DCCA
        dcca_model = DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2],
                          objective=objectives.CCA)
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dcca_model = DeepWrapper(dcca_model, device=device)
        dcca_model.fit((self.X, self.Y), epochs=10)
        # DGCCA
        dgcca_model = DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2],
                           objective=objectives.GCCA)
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dgcca_model = DeepWrapper(dgcca_model, device=device)
        dgcca_model.fit((self.X, self.Y), epochs=3)
        # DMCCA
        dmcca_model = DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2],
                           objective=objectives.MCCA)
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dmcca_model = DeepWrapper(dmcca_model, device=device)
        dmcca_model.fit((self.X, self.Y), epochs=3)
        # DCCA_NOI
        dcca_noi_model = DCCA_NOI(latent_dims=latent_dims, encoders=[encoder_1, encoder_2])
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dcca_noi_model = DeepWrapper(dcca_noi_model, device=device)
        dcca_noi_model.fit((self.X, self.Y), epochs=30)

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
        dcca_model = DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2, encoder_3],
                          objective=objectives.GCCA)
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dcca_model = DeepWrapper(dcca_model, device=device)
        dcca_model.fit((self.X, self.Y, self.Z), epochs=20)
        print('here')

    def test_schedulers(self):
        latent_dims = 2
        device = 'cpu'
        encoder_1 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
        encoder_2 = architectures.Encoder(latent_dims=latent_dims, feature_size=10)
        # DCCA
        optimizers = [optim.Adam(encoder_1.parameters(), lr=1e-4), optim.Adam(encoder_2.parameters(), lr=1e-4)]
        schedulers = [optim.lr_scheduler.CosineAnnealingLR(optimizers[0], 1),
                      optim.lr_scheduler.ReduceLROnPlateau(optimizers[1])]
        dcca_model = DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2],
                          objective=objectives.CCA, optimizers=optimizers, schedulers=schedulers)
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dcca_model = DeepWrapper(dcca_model, device=device)
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
                            decoders=[decoder_1, decoder_2], objective=objectives.CCA)
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dccae_model = DeepWrapper(dccae_model, device=device)
        dccae_model.fit((self.X, self.Y))
        # SplitAE
        splitae_model = SplitAE(latent_dims=latent_dims, encoder=encoder_1,
                                decoders=[decoder_1, decoder_2])
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        splitae_model = DeepWrapper(splitae_model, device=device)
        splitae_model.fit((self.X, self.Y), train_correlations=False)

    def test_DCCAEconv_methods_cpu(self):
        latent_dims = 2
        device = 'cpu'
        encoder_1 = architectures.CNNEncoder(latent_dims=latent_dims, feature_size=[16, 16])
        encoder_2 = architectures.CNNEncoder(latent_dims=latent_dims, feature_size=[16, 16])
        decoder_1 = architectures.CNNDecoder(latent_dims=latent_dims, feature_size=[16, 16])
        decoder_2 = architectures.CNNDecoder(latent_dims=latent_dims, feature_size=[16, 16])
        # DCCAE
        dccae_model = DCCAE(latent_dims=latent_dims, encoders=[encoder_1, encoder_2],
                            decoders=[decoder_1, decoder_2], objective=objectives.CCA)
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
        dvcca_model = DVCCA(latent_dims=latent_dims, private=True, encoders=[encoder_1, encoder_2],
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
        dcca_noi_model = DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2])
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
                            decoders=[decoder_1, decoder_2], objective=objectives.CCA)
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
