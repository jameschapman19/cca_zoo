import numpy as np
from unittest import TestCase
import cca_zoo.deepwrapper
import cca_zoo.configuration


class TestDeepWrapper(TestCase):

    def setUp(self):
        self.X = np.random.rand(30, 10)
        self.Y = np.random.rand(30, 10)
        self.Z = np.random.rand(30, 10)
        self.cfg = cca_zoo.configuration.Config()

    def tearDown(self):
        pass

    def test_DCCA_methods(self):
        self.cfg = cca_zoo.configuration.Config()
        dcca = cca_zoo.deepwrapper.DeepWrapper(self.cfg)
        dcca.fit(self.X, self.Y)
        self.cfg.loss_type = cca_zoo.objectives.GCCA
        dgcca = cca_zoo.deepwrapper.DeepWrapper(self.cfg)
        dgcca.fit(self.X, self.Y)
        self.cfg.loss_type = cca_zoo.objectives.MCCA
        dmcca = cca_zoo.deepwrapper.DeepWrapper(self.cfg)
        dmcca.fit(self.X, self.Y)
        self.cfg.als = True
        dcca_als = cca_zoo.deepwrapper.DeepWrapper(self.cfg)
        dcca_als.fit(self.X, self.Y)

    def test_DGCCA_methods(self):
        self.cfg = cca_zoo.configuration.Config()
        self.cfg.latent_dims = 2
        self.cfg.epoch_num = 20
        self.cfg.encoder_models = [cca_zoo.deep_models.Encoder, cca_zoo.deep_models.Encoder,
                                   cca_zoo.deep_models.Encoder]
        self.cfg.hidden_layer_sizes = [[128], [128], [128]]
        self.cfg.objective = cca_zoo.objectives.MCCA
        dmcca = cca_zoo.deepwrapper.DeepWrapper(self.cfg)
        dmcca.fit(self.X, self.Y, self.Z)
        self.cfg.objective = cca_zoo.objectives.GCCA
        dgcca = cca_zoo.deepwrapper.DeepWrapper(self.cfg)
        dgcca.fit(self.X, self.Y, self.Z)


    def test_DCCAE_methods(self):
        self.cfg = cca_zoo.configuration.Config()
        self.cfg.method = cca_zoo.dccae.DCCAE
        dccae = cca_zoo.deepwrapper.DeepWrapper(self.cfg)
        dccae.fit(self.X, self.Y)

    def test_DVCCA_methods(self):
        self.cfg = cca_zoo.configuration.Config()
        self.cfg.method = cca_zoo.dvcca.DVCCA
        dvcca = cca_zoo.deepwrapper.DeepWrapper(self.cfg)
        dvcca.fit(self.X, self.Y)
