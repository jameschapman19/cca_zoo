from unittest import TestCase
from torch import optim
import numpy as np

import cca_zoo.dcca
import cca_zoo.dccae
import cca_zoo.deep_models
import cca_zoo.deepwrapper
import cca_zoo.dvcca


class TestDeepWrapper(TestCase):

    def setUp(self):
        self.X = np.random.rand(30, 10)
        self.Y = np.random.rand(30, 10)
        self.Z = np.random.rand(30, 10)

    def tearDown(self):
        pass

    def test_DCCA_methods_cpu(self):
        latent_dims = 2
        device = 'cpu'
        encoder_1 = cca_zoo.deep_models.Encoder(latent_dims=latent_dims, feature_size=10)
        encoder_2 = cca_zoo.deep_models.Encoder(latent_dims=latent_dims, feature_size=10)
        # DCCA
        dcca_model = cca_zoo.dcca.DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2],
                                       objective=cca_zoo.objectives.CCA)
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dcca_model = cca_zoo.deepwrapper.DeepWrapper(dcca_model, device=device)
        dcca_model.fit(self.X, self.Y)
        # DGCCA
        dgcca_model = cca_zoo.dcca.DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2],
                                        objective=cca_zoo.objectives.GCCA)
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dgcca_model = cca_zoo.deepwrapper.DeepWrapper(dgcca_model, device=device)
        dgcca_model.fit(self.X, self.Y)
        # DMCCA
        dmcca_model = cca_zoo.dcca.DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2],
                                        objective=cca_zoo.objectives.MCCA)
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dmcca_model = cca_zoo.deepwrapper.DeepWrapper(dmcca_model, device=device)
        dmcca_model.fit(self.X, self.Y)
        # DCCA_NOI
        dcca_noi_model = cca_zoo.dcca.DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2], als=True)
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dcca_noi_model = cca_zoo.deepwrapper.DeepWrapper(dcca_noi_model, device=device)
        dcca_noi_model.fit(self.X, self.Y)

    def test_schedulers(self):
        latent_dims = 2
        device = 'cpu'
        encoder_1 = cca_zoo.deep_models.Encoder(latent_dims=latent_dims, feature_size=10)
        encoder_2 = cca_zoo.deep_models.Encoder(latent_dims=latent_dims, feature_size=10)
        # DCCA
        optimizers = [optim.Adam(encoder_1.parameters(), lr=1e-4), optim.Adam(encoder_2.parameters(), lr=1e-4)]
        schedulers = [optim.lr_scheduler.CosineAnnealingLR(optimizers[0], 1),
                      optim.lr_scheduler.ReduceLROnPlateau(optimizers[1])]
        dcca_model = cca_zoo.dcca.DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2],
                                       objective=cca_zoo.objectives.CCA, optimizers=optimizers, schedulers=schedulers)
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dcca_model = cca_zoo.deepwrapper.DeepWrapper(dcca_model, device=device)
        dcca_model.fit(self.X, self.Y,epochs=20)

    def test_DGCCA_methods_cpu(self):
        latent_dims = 2
        device = 'cpu'
        encoder_1 = cca_zoo.deep_models.Encoder(latent_dims=latent_dims, feature_size=10)
        encoder_2 = cca_zoo.deep_models.Encoder(latent_dims=latent_dims, feature_size=10)
        encoder_3 = cca_zoo.deep_models.Encoder(latent_dims=latent_dims, feature_size=10)
        # DGCCA
        dgcca_model = cca_zoo.dcca.DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2, encoder_3],
                                        objective=cca_zoo.objectives.GCCA)
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dgcca_model = cca_zoo.deepwrapper.DeepWrapper(dgcca_model, device=device)
        dgcca_model.fit(self.X, self.Y, self.Z)
        # DMCCA
        dmcca_model = cca_zoo.dcca.DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2, encoder_3],
                                        objective=cca_zoo.objectives.MCCA)
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dmcca_model = cca_zoo.deepwrapper.DeepWrapper(dmcca_model, device=device)
        dmcca_model.fit(self.X, self.Y, self.Z)

    def test_DCCAE_methods_cpu(self):
        latent_dims = 2
        device = 'cpu'
        encoder_1 = cca_zoo.deep_models.Encoder(latent_dims=latent_dims, feature_size=10)
        encoder_2 = cca_zoo.deep_models.Encoder(latent_dims=latent_dims, feature_size=10)
        decoder_1 = cca_zoo.deep_models.Decoder(latent_dims=latent_dims, feature_size=10)
        decoder_2 = cca_zoo.deep_models.Decoder(latent_dims=latent_dims, feature_size=10)
        # DCCAE
        dccae_model = cca_zoo.dccae.DCCAE(latent_dims=latent_dims, encoders=[encoder_1, encoder_2],
                                          decoders=[decoder_1, decoder_2], objective=cca_zoo.objectives.CCA)
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dccae_model = cca_zoo.deepwrapper.DeepWrapper(dccae_model, device=device)
        dccae_model.fit(self.X, self.Y)

    def test_DVCCA_methods_cpu(self):
        latent_dims = 2
        device = 'cpu'
        encoder_1 = cca_zoo.deep_models.Encoder(latent_dims=latent_dims, feature_size=10, variational=True)
        encoder_2 = cca_zoo.deep_models.Encoder(latent_dims=latent_dims, feature_size=10, variational=True)
        decoder_1 = cca_zoo.deep_models.Decoder(latent_dims=latent_dims, feature_size=10, norm_output=True)
        decoder_2 = cca_zoo.deep_models.Decoder(latent_dims=latent_dims, feature_size=10, norm_output=True)
        # DVCCA
        dvcca_model = cca_zoo.dvcca.DVCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2],
                                          decoders=[decoder_1, decoder_2])
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dvcca_model = cca_zoo.deepwrapper.DeepWrapper(dvcca_model, device=device)
        dvcca_model.fit(self.X, self.Y)

        private_encoder_1 = cca_zoo.deep_models.Encoder(latent_dims=latent_dims, feature_size=10, variational=True)
        private_encoder_2 = cca_zoo.deep_models.Encoder(latent_dims=latent_dims, feature_size=10, variational=True)
        # DVCCA
        dvcca_model = cca_zoo.dvcca.DVCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2],
                                          decoders=[decoder_1, decoder_2],private_encoders=[private_encoder_1,private_encoder_2])
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dvcca_model = cca_zoo.deepwrapper.DeepWrapper(dvcca_model, device=device)
        dvcca_model.fit(self.X, self.Y)

    def test_DCCA_methods_gpu(self):
        latent_dims = 2
        device = 'cuda'
        encoder_1 = cca_zoo.deep_models.Encoder(latent_dims=latent_dims, feature_size=10)
        encoder_2 = cca_zoo.deep_models.Encoder(latent_dims=latent_dims, feature_size=10)
        # DCCA
        dcca_model = cca_zoo.dcca.DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2],
                                       objective=cca_zoo.objectives.CCA)
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dcca_model = cca_zoo.deepwrapper.DeepWrapper(dcca_model, device=device)
        dcca_model.fit(self.X, self.Y)
        # DGCCA
        dgcca_model = cca_zoo.dcca.DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2],
                                        objective=cca_zoo.objectives.GCCA)
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dgcca_model = cca_zoo.deepwrapper.DeepWrapper(dgcca_model, device=device)
        dgcca_model.fit(self.X, self.Y)
        # DMCCA
        dmcca_model = cca_zoo.dcca.DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2],
                                        objective=cca_zoo.objectives.MCCA)
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dmcca_model = cca_zoo.deepwrapper.DeepWrapper(dmcca_model, device=device)
        dmcca_model.fit(self.X, self.Y)
        # DCCA_NOI
        dcca_noi_model = cca_zoo.dcca.DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2], als=True)
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dcca_noi_model = cca_zoo.deepwrapper.DeepWrapper(dcca_noi_model, device=device)
        dcca_noi_model.fit(self.X, self.Y)

    def test_DGCCA_methods_gpu(self):
        latent_dims = 2
        device = 'cuda'
        encoder_1 = cca_zoo.deep_models.Encoder(latent_dims=latent_dims, feature_size=10)
        encoder_2 = cca_zoo.deep_models.Encoder(latent_dims=latent_dims, feature_size=10)
        encoder_3 = cca_zoo.deep_models.Encoder(latent_dims=latent_dims, feature_size=10)
        # DGCCA
        dgcca_model = cca_zoo.dcca.DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2, encoder_3],
                                        objective=cca_zoo.objectives.GCCA)
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dgcca_model = cca_zoo.deepwrapper.DeepWrapper(dgcca_model, device=device)
        dgcca_model.fit(self.X, self.Y, self.Z)
        # DMCCA
        dmcca_model = cca_zoo.dcca.DCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2, encoder_3],
                                        objective=cca_zoo.objectives.MCCA)
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dmcca_model = cca_zoo.deepwrapper.DeepWrapper(dmcca_model, device=device)
        dmcca_model.fit(self.X, self.Y, self.Z)

    def test_DCCAE_methods_gpu(self):
        latent_dims = 2
        device = 'cuda'
        encoder_1 = cca_zoo.deep_models.Encoder(latent_dims=latent_dims, feature_size=10)
        encoder_2 = cca_zoo.deep_models.Encoder(latent_dims=latent_dims, feature_size=10)
        decoder_1 = cca_zoo.deep_models.Decoder(latent_dims=latent_dims, feature_size=10)
        decoder_2 = cca_zoo.deep_models.Decoder(latent_dims=latent_dims, feature_size=10)
        # DCCAE
        dccae_model = cca_zoo.dccae.DCCAE(latent_dims=latent_dims, encoders=[encoder_1, encoder_2],
                                          decoders=[decoder_1, decoder_2], objective=cca_zoo.objectives.CCA)
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dccae_model = cca_zoo.deepwrapper.DeepWrapper(dccae_model, device=device)
        dccae_model.fit(self.X, self.Y)

    def test_DVCCA_methods_gpu(self):
        latent_dims = 2
        device = 'cuda'
        encoder_1 = cca_zoo.deep_models.Encoder(latent_dims=latent_dims, feature_size=10, variational=True)
        encoder_2 = cca_zoo.deep_models.Encoder(latent_dims=latent_dims, feature_size=10, variational=True)
        decoder_1 = cca_zoo.deep_models.Decoder(latent_dims=latent_dims, feature_size=10, norm_output=True)
        decoder_2 = cca_zoo.deep_models.Decoder(latent_dims=latent_dims, feature_size=10, norm_output=True)
        # DVCCA
        dvcca_model = cca_zoo.dvcca.DVCCA(latent_dims=latent_dims, encoders=[encoder_1, encoder_2],
                                          decoders=[decoder_1, decoder_2])
        # hidden_layer_sizes are shown explicitly but these are also the defaults
        dvcca_model = cca_zoo.deepwrapper.DeepWrapper(dvcca_model, device=device)
        dvcca_model.fit(self.X, self.Y)
