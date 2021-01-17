.. cca-zoo documentation master file, created by
   sphinx-quickstart on Wed Dec  2 17:53:47 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Deep Models
===========

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Intro
-----

DeepWrapper
-----------

The deepwrapper provides a general training scheme.

.. automodule:: cca_zoo.deepwrapper
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Config
------

The main difference between running linear models and deep learning based models is the Config class from configuration.py.

I introduced Config in order to allow better flexibility of encoder and decoder architectures within broader model architectures
(Deep Canonical Correlation Analysis, Deep Canonically Correlated Autoencoders, Deep Variational Canonical Correlation Analysis).

Config() holds default settings such that all 3 model architectures will work out of the box but I will try to demonstrate
how one might change these defaults.

.. sourcecode:: python

   class Config:
       def __init__(self):
           #Defines the basic architecture DCCA, DCCAE, DVCCA
           self.method = cca_zoo.dcca.DCCA
           #The number of encoding dimensions
           self.latent_dims = 2
           self.learning_rate = 1e-3
           self.epoch_num = 1
           self.patience = 0
           self.batch_size = 0
           #Updated automatically when using deepwrapper.DeepWrapper
           self.input_sizes = None
           #These control the encoder architectures. We need one for each view. Fully connected models provided by default
           self.encoder_models = [cca_zoo.deep_models.Encoder, cca_zoo.deep_models.Encoder]
           # These control the decoder architectures. We need one for each view if using DCCAE or DVCCA. Fully connected models provided by default
           self.decoder_models = [cca_zoo.deep_models.Decoder, cca_zoo.deep_models.Decoder]
           #These are parameters used by cca_zoo.deep_models.Encoder
           self.hidden_layer_sizes = [[128], [128]]
           # We can choose to use cca_zoo.objectives.CCA, cca_zoo.objectives.MCCA, cca_zoo.objectives.GCCA
           self.objective = cca_zoo.objectives.CCA
           # We also implement DCCA by non-linear orthogonal iterations (alternating least squares).
           self.als = False
           self.eps = 1e-9

           #Used for DCCAE:
           # Weighting of reconstruction vs correlation loss
           self.lam = 0

           # Used for DVCCA:
           # True gives bi-DVCCA, False gives DVCCA
           self.both_encoders = True
           # True gives DVCCA_private, False gives DVCCA
           self.private = False
           # mu from the original paper controls the weighting of each encoder
           self.mu = 0.5

Models Implemented
------------------

.. toctree::
   :maxdepth: 3

   deepcca.rst
   deepccae.rst
   deepvcca.rst

Architecture Options
--------------------

We provide a base encoder and decoder class which can be inherited in order to build custom architectures which can be
used

.. automodule:: cca_zoo.deep_models
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource