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
        # Defines the basic architecture DCCA, DCCAE, DVCCA
        self.device = None
        self.method = cca_zoo.dcca.DCCA
        # The number of encoding dimensions
        self.latent_dims = 2
        self.learning_rate = 1e-3
        self.epoch_num = 1
        self.patience = 0
        self.batch_size = 0
        self.eps = 1e-9
        self.c = 1e-5
        # Updated automatically when using deepwrapper.DeepWrapper
        self.input_sizes = None
        # These control the encoder architectures. We need one for each view. Fully connected models provided by default
        self.encoder_models = [cca_zoo.deep_models.Encoder, cca_zoo.deep_models.Encoder]
        self.encoder_args = [{'layer_sizes': [256]}, {'layer_sizes': [256]}]
        # These control the decoder architectures. We need one for each view if using DCCAE or DVCCA. Fully connected models provided by default
        self.decoder_models = [cca_zoo.deep_models.Decoder, cca_zoo.deep_models.Decoder]
        self.decoder_args = [{'layer_sizes': [256]}, {'layer_sizes': [256]}]

        # We can choose to use cca_zoo.objectives.CCA, cca_zoo.objectives.MCCA, cca_zoo.objectives.GCCA
        self.objective = cca_zoo.objectives.CCA
        # We also implement DCCA by non-linear orthogonal iterations (alternating least squares).
        self.als = False
        self.rho = 0.2

        # Learn linear CCA to transform data mapped by tracenorm objective
        self.post_transform = True

        # Used for DCCAE:
        # Weighting of reconstruction vs correlation loss
        self.lam = 0.5

        # Used for DVCCA:
        self.private_encoder_models = [cca_zoo.deep_models.Encoder, cca_zoo.deep_models.Encoder]
        self.private_encoder_args = [{'layer_sizes': [256]}, {'layer_sizes': [256]}]
        # True gives DVCCA_private, False gives DVCCA
        self.private = False
        # mu from the original paper controls the weighting of each encoder
        self.mu = 0.5

        # Not used yet
        self.autoencoder = False
        self.confound_encoder_models = [cca_zoo.deep_models.Encoder]
        self.confound_encoder_args = [{'layer_sizes': [256]}]
