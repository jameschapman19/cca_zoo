import cca_zoo.dcca
import cca_zoo.deep_models
import cca_zoo.objectives

"""
The intention is that this configuration class works as both default parameters
as well as something that can be edited to use different deep methods or 
encoder/decoder models in the DCCA. I think I will do something similar for the linear models.
"""


class Config:
    def __init__(self):
        self.rho = 0.75
        self.als = False
        self.eps = 1e-9
        self.lamda = 100
        self.method = cca_zoo.dcca.DCCA
        self.mu = 0.5
        self.objective = cca_zoo.objectives.CCA
        self.latent_dims = 2
        self.learning_rate = 1e-3
        self.epoch_num = 1
        self.batch_size = 0
        self.both_encoders = True
        self.private = False
        self.patience = 0
        self.loss_type = 'cca'
        self.lam = 0
        self.encoder_models = [cca_zoo.deep_models.Encoder, cca_zoo.deep_models.Encoder]
        self.decoder_models = [cca_zoo.deep_models.Decoder, cca_zoo.deep_models.Decoder]
        self.confound_encoder_models = [cca_zoo.deep_models.Encoder]
        self.hidden_layer_sizes = [[128], [128]]
        self.input_sizes = None
